# ==============================================================================
# BTC QUANT GODMODE PRO MAX ENGINE - HONEST VERSION (TIMEFRAME 6 JAM)
# Fitur: Anti-Repainting, Zero Data Leakage, Crypto-NLP, Fixed Kelly, Indodax
# Visual: Grafik Jujur (Berdasarkan Log CSV), Dual Chart, USD Stabil
# Zona Waktu: WITA (Asia/Makassar)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import requests
import os
import warnings
import urllib.request
import xml.etree.ElementTree as ET
import re
import time
import csv
from datetime import datetime

# Import Library ML dan Finance
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Mematikan warning yang tidak perlu agar log GitHub bersih
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 

# ================= KONFIGURASI TELEGRAM & SERVER =================
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# File untuk mencatat sejarah prediksi (Anti-Repainting)
HISTORY_FILE = "ai_history_log.csv"
# ==================================================================

def format_rupiah(angka):
    if pd.isna(angka): return "Rp 0"
    return f"Rp {angka:,.0f}".replace(',', '.')

def format_usd(angka):
    if pd.isna(angka): return "$0"
    return f"${angka:,.0f}"

# ------------------------------------------------------------------------------
# 1. MODUL SENTIMEN BERITA (VADER + CRYPTO LEXICON)
# ------------------------------------------------------------------------------
def fetch_crypto_news_sentiment():
    print("[*] Menganalisis berita Kripto dengan NLP (Crypto-Tuned)...")
    rss_urls = [
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://cointelegraph.com/rss'
    ]
    
    analyzer = SentimentIntensityAnalyzer()
    crypto_lexicon = {
        "bullish": 2.5, "bearish": -2.5, "rekt": -3.0, "moon": 2.5, 
        "pump": 2.0, "dump": -2.5, "fud": -2.0, "hack": -3.0, "scam": -3.0,
        "ath": 2.0, "approved": 2.0, "banned": -2.5
    }
    analyzer.lexicon.update(crypto_lexicon)
    
    compound_scores = []
    unique_news = set()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36'
    }
    
    for url in rss_urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall('.//item')[:10]:
                    title = item.find('title').text
                    if not title: continue
                    
                    clean_title = re.sub(r'[^\w\s]', '', title.lower())
                    if clean_title not in unique_news:
                        unique_news.add(clean_title)
                        sentiment = analyzer.polarity_scores(title)
                        compound_scores.append(sentiment['compound'])
        except:
            continue

    if not compound_scores: return "TIDAK ADA DATA BERITA ⚪"
    avg_score = sum(compound_scores) / len(compound_scores)
    
    if avg_score >= 0.25: return f"SANGAT POSITIF 🚀 ({len(unique_news)} Berita)"
    elif avg_score > 0.05: return f"POSITIF RINGAN 🟢 ({len(unique_news)} Berita)"
    elif avg_score <= -0.25: return f"SANGAT NEGATIF 🚨 ({len(unique_news)} Berita)"
    elif avg_score < -0.05: return f"NEGATIF RINGAN 🔴 ({len(unique_news)} Berita)"
    return f"NETRAL/SEIMBANG ⚪ ({len(unique_news)} Berita)"

# ------------------------------------------------------------------------------
# 2. MODUL DATA (GLOBAL USD UNTUK ML, INDODAX UNTUK LAPORAN)
# ------------------------------------------------------------------------------
# [PERBAIKAN SWEET SPOT] Menggunakan period 90 Hari (Cukup cerdas untuk AI, 
# tapi cukup ringan agar server Yahoo tidak nyangkut parah).
def fetch_and_engineer_features(period='90d', interval='1h'):
    print("[*] Mengunduh data pasar Global (Bebas Distorsi Indodax)...")
    df = yf.download('BTC-USD', period=period, interval=interval, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.droplevel(1)

    if df.index.tzinfo is None: 
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Asia/Makassar') # WITA

    try:
        idr_data = yf.download('IDR=X', period='5d', progress=False)
        if isinstance(idr_data.columns, pd.MultiIndex): 
            idr_data.columns = idr_data.columns.droplevel(1)
        kurs_idr = float(idr_data['Close'].dropna().iloc[-1])
    except:
        kurs_idr = 16000.0

    indodax_live_idr = 0.0
    try:
        indodax_req = requests.get('https://indodax.com/api/ticker/btcidr', timeout=5).json()
        indodax_live_idr = float(indodax_req['ticker']['last'])
    except:
        indodax_live_idr = float(df['Close'].iloc[-1]) * kurs_idr

    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_20 * 2.0)
    df['BB_Lower'] = df['BB_Middle'] - (std_20 * 2.0)
    
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))
    df['StochRSI'] = (df['RSI'] - df['RSI'].rolling(14).min()) / (df['RSI'].rolling(14).max() - df['RSI'].rolling(14).min())

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1).rolling(14).mean()
    
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['ADX'] = (100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])).ewm(alpha=1/14, adjust=False).mean()

    if 'Volume' in df.columns:
        df['VP'] = ((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']
        df['VWAP_24'] = df['VP'].rolling(window=24).sum() / df['Volume'].rolling(window=24).sum()
    else:
        df['VWAP_24'] = df['Close']

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA_Spread"] = (df["EMA20"] - df["EMA50"]) / df["Close"]
    
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    df["MACD_Hist"] = macd - macd.ewm(span=9).mean()
    
    df["Return_1H"] = df["Close"].pct_change()
    df["Return_6H"] = df["Close"].pct_change(6)
    df["Volatility"] = df["Return_1H"].rolling(24).std()
    df["Trend_Slope"] = df["Close"].rolling(12).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)==12 else 0, raw=True)
    df["Momentum_Accel"] = df["Return_1H"].diff()
    df["Regime"] = np.where(df["EMA20"] > df["EMA50"], 1, 0)
    
    df["Target"] = (df["Close"].shift(-6) > df["Close"]).astype(float)
    
    df = df.iloc[:-1]
    
    features_cols = ["EMA_Spread","RSI","MACD_Hist","Return_1H","Return_6H",
                     "Volatility","Trend_Slope","Momentum_Accel","Regime"]
    
    return df, features_cols, kurs_idr, indodax_live_idr

# ------------------------------------------------------------------------------
# 3. MODUL AI JUJUR (TANPA DATA LEAKAGE & REPAINTING)
# ------------------------------------------------------------------------------
def train_and_predict_honest(df, features):
    print("[*] Melatih AI Ensemble (Zero Leakage Pipeline)...")
    
    train_df = df.iloc[:-6].dropna(subset=features + ["Target"])
    X_train = train_df[features].values
    y_train = train_df["Target"].values
    
    X_live = df[features].iloc[-1:].fillna(0).values

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    lr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    
    for train_index, test_index in tscv.split(X_train):
        X_tr, X_te = X_train[train_index], X_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        
        fold_scaler = StandardScaler()
        X_tr_scaled = fold_scaler.fit_transform(X_tr)
        X_te_scaled = fold_scaler.transform(X_te)
        
        lr.fit(X_tr_scaled, y_tr)
        rf.fit(X_tr_scaled, y_tr)
        gb.fit(X_tr_scaled, y_tr)
        
        p_lr = lr.predict_proba(X_te_scaled)[:, 1]
        p_rf = rf.predict_proba(X_te_scaled)[:, 1]
        p_gb = gb.predict_proba(X_te_scaled)[:, 1]
        
        ensemble_cv_prob = (p_lr + p_rf + p_gb) / 3
        fold_acc = accuracy_score(y_te, (ensemble_cv_prob > 0.5).astype(int))
        cv_scores.append(fold_acc)
        
    accuracy = np.mean(cv_scores) * 100

    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_live_scaled = final_scaler.transform(X_live)

    lr_cal = CalibratedClassifierCV(LogisticRegression(), method="sigmoid", cv=3)
    lr_cal.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)

    prob_lr = lr_cal.predict_proba(X_live_scaled)[0,1]
    prob_rf = rf.predict_proba(X_live_scaled)[0,1]
    prob_gb = gb.predict_proba(X_live_scaled)[0,1]
    
    latest_prob = (prob_lr + prob_rf + prob_gb) / 3

    return latest_prob, accuracy

# ------------------------------------------------------------------------------
# 3.5 MODUL DATABASE LOKAL (ANTI-REPAINTING SYSTEM)
# ------------------------------------------------------------------------------
def save_and_load_predictions(df, latest_prob):
    print("[*] Sinkronisasi Jurnal Prediksi Lokal (Mencegah Repainting)...")
    
    current_time = df.index[-1]
    current_price = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    predicted_target = current_price + (atr * (latest_prob - 0.5) * 4)
    next_time_target = current_time + pd.Timedelta(hours=6) 
    
    file_exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp_target', 'predicted_price', 'prob_up'])
        writer.writerow([next_time_target.isoformat(), predicted_target, latest_prob])

    df['Honest_Prediction'] = np.nan
    eval_msg = "Menunggu data 6 jam ke depan untuk evaluasi..."
    
    if file_exists:
        try:
            history = pd.read_csv(HISTORY_FILE)
            history['timestamp_target'] = pd.to_datetime(history['timestamp_target'], utc=True).dt.tz_convert('Asia/Makassar')
            
            for _, row in history.iterrows():
                ts = row['timestamp_target']
                if ts in df.index:
                    df.loc[ts, 'Honest_Prediction'] = row['predicted_price']
            
            if current_time in df.index and len(history) >= 2:
                past_pred_row = history[history['timestamp_target'] == current_time]
                if not past_pred_row.empty:
                    past_prob = past_pred_row['prob_up'].values[0]
                    
                    if len(df) > 7:
                        prev_6h_close = df['Close'].iloc[-7]
                        aktual_sekarang = df['Close'].iloc[-1]
                        
                        arah_prediksi = "NAIK" if past_prob > 0.5 else "TURUN"
                        arah_aktual = "Naik" if aktual_sekarang > prev_6h_close else "Turun"
                        
                        if (past_prob > 0.5 and aktual_sekarang > prev_6h_close) or (past_prob <= 0.5 and aktual_sekarang <= prev_6h_close):
                            eval_msg = f"BENAR ✅ (6 jam lalu AI menebak {arah_prediksi}, harga asli {arah_aktual})"
                        else:
                            eval_msg = f"SALAH ❌ (6 jam lalu AI menebak {arah_prediksi}, harga malah {arah_aktual})"
        except Exception as e:
            print(f"[!] Gagal membaca log history: {e}")

    return df, predicted_target, next_time_target, eval_msg

# ------------------------------------------------------------------------------
# 4. MODUL MONTE CARLO & DYNAMIC KELLY CRITERION (FIXED MATH)
# ------------------------------------------------------------------------------
def monte_carlo_simulation(price, vol, steps=24, sims=2000):
    paths = []
    for _ in range(sims):
        prices = [price]
        for _ in range(steps):
            shock = np.random.normal(0, vol)
            prices.append(prices[-1] * (1 + shock))
        paths.append(prices)

    paths = np.array(paths)
    final_prices = paths[:, -1]
    
    expected = np.mean(final_prices)
    var95 = np.percentile(final_prices, 5) 
    return expected, var95

def position_sizing_kelly(prob, atr):
    if prob < 0.5:
        return 0.0
        
    reward = atr * 3.5 
    risk = atr * 1.5
    
    if risk <= 0: return 0.0 
    
    r_ratio = reward / risk
    edge = prob - ((1 - prob) / r_ratio)
    kelly = max(edge, 0)
    
    size = min(kelly, 0.25) 
    return round(size * 100, 2)

# ------------------------------------------------------------------------------
# 5A. VISUALISASI CHART UTAMA (HONEST PLOT)
# ------------------------------------------------------------------------------
def plot_professional_analysis(df, next_time, next_pred, filename="chart_main.png"):
    plot_data = df.tail(80) 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    ax1.plot(plot_data.index, plot_data['Close'], label='Harga Asli BTC (USD)', color='black', linewidth=2, zorder=5)
    ax1.plot(plot_data.index, plot_data['VWAP_24'], label='Garis Bandar (VWAP)', color='#ff7f0e', linestyle='-.', linewidth=2)
    ax1.plot(plot_data.index, plot_data['MA_50'], label='Trend (MA50)', color='blue', alpha=0.6)
    
    if 'Honest_Prediction' in plot_data.columns and plot_data['Honest_Prediction'].notna().any():
        valid_preds = plot_data.dropna(subset=['Honest_Prediction'])
        if not valid_preds.empty:
            ax1.plot(valid_preds.index, valid_preds['Honest_Prediction'], 
                     label='Jejak Asli AI (Tanpa Repaint)', color='red', linestyle='--', linewidth=2, alpha=0.8)

    last_time = plot_data.index[-1]
    ax1.scatter(next_time, next_pred, color='red', s=80, marker='o', zorder=10, label=f'Target 6 Jam Depan')
    
    if 'Honest_Prediction' in df.columns and pd.notna(df['Honest_Prediction'].iloc[-1]):
        start_price = df['Honest_Prediction'].iloc[-1]
    else:
        start_price = df['Close'].iloc[-1]
    ax1.plot([last_time, next_time], [start_price, next_pred], color='red', linestyle='--', linewidth=2)

    ax1.set_title('GRAFIK UTAMA AI (ANTI-REPAINTING) - 80 JAM', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Harga (USD)')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    
    myFmt = mdates.DateFormatter('%d %b\n%H:%M')
    
    ax2.plot(plot_data.index, plot_data['RSI'], color='purple', label='RSI (Momentum)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')

    ax3.plot(plot_data.index, plot_data['ADX'], color='brown', linewidth=2, label='ADX (Tren)')
    ax3.axhline(25, color='black', linestyle='--', alpha=0.8)
    ax3.set_ylabel('ADX')
    ax3.set_ylim(0, 60)
    ax3.legend(loc='upper left')

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(myFmt)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 5B. VISUALISASI CHART ZOOM (LABEL HARGA)
# ------------------------------------------------------------------------------
def plot_zoomed_analysis(df, next_time, next_pred, filename="chart_zoom.png"):
    plot_data = df.tail(12) 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(plot_data.index, plot_data['Close'], label='Harga Asli BTC (USD)', color='black', linewidth=3, zorder=5)
    ax1.plot(plot_data.index, plot_data['VWAP_24'], label='Garis Bandar (VWAP)', color='#ff7f0e', linestyle='-.', linewidth=2)
    ax1.fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], color='gray', alpha=0.15)

    last_time = plot_data.index[-1]
    last_price_usd = plot_data['Close'].iloc[-1]
    
    if 'Honest_Prediction' in plot_data.columns:
        valid_preds = plot_data.dropna(subset=['Honest_Prediction'])
        if not valid_preds.empty:
            ax1.plot(valid_preds.index, valid_preds['Honest_Prediction'], 
                     label='Jejak AI Masa Lalu', color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    
    if 'Honest_Prediction' in df.columns and pd.notna(df['Honest_Prediction'].iloc[-1]):
        start_price = df['Honest_Prediction'].iloc[-1]
    else:
        start_price = last_price_usd
        
    ax1.scatter(next_time, next_pred, color='red', s=120, marker='o', zorder=10)
    ax1.plot([last_time, next_time], [start_price, next_pred], color='red', linestyle='--', linewidth=2.5)

    ax1.annotate(f"Harga Asli: {format_usd(last_price_usd)}",
                 (last_time, last_price_usd),
                 xytext=(0, -25), textcoords='offset points', ha='center',
                 bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="none", alpha=0.8),
                 color="white", fontweight='bold', fontsize=11, zorder=15)
                 
    ax1.annotate(f"Target AI: {format_usd(next_pred)}",
                 (next_time, next_pred),
                 xytext=(0, 15), textcoords='offset points', ha='center',
                 bbox=dict(boxstyle="round,pad=0.4", fc="red", ec="none", alpha=0.9),
                 color="white", fontweight='bold', fontsize=11, zorder=15)

    ax1.set_title('🔍 ZOOM TERAKHIR & ARAH TARGET BERIKUTNYA', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Harga (USD)')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=11)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 6. MODUL TELEGRAM 
# ------------------------------------------------------------------------------
def send_to_telegram(message, image_paths):
    print("[*] Mengirim laporan ke Telegram...")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Pengiriman dibatalkan. Token Telegram belum disetel.")
        return
        
    url_message = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url_message, data={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'})

    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    for path in image_paths:
        if os.path.exists(path):
            with open(path, 'rb') as photo:
                requests.post(url_photo, data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': photo})
    print("[*] Selesai Mengirim!")

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    start_time = time.time()
    print("==========================================================")
    print("   QUANT GODMODE ENGINE - HONEST & LEAKAGE-FREE VERSION   ")
    print("==========================================================")
    
    df, features, kurs_idr, indodax_live_idr = fetch_and_engineer_features()
    news_status = fetch_crypto_news_sentiment()
    latest_close_usd = df['Close'].iloc[-1]
    volatility = df["Volatility"].iloc[-1]
    current_atr = df["ATR"].iloc[-1]
    
    latest_prob, ml_accuracy = train_and_predict_honest(df, features)
    
    df, next_target_usd, next_time, eval_msg = save_and_load_predictions(df, latest_prob)
    
    exposure = position_sizing_kelly(latest_prob, current_atr)
    
    expected_24h_usd, var95_usd = monte_carlo_simulation(latest_close_usd, volatility)
    expected_24h_idr = expected_24h_usd * kurs_idr
    var95_idr = var95_usd * kurs_idr
    
    waktu_eksekusi = time.time() - start_time
    waktu_data_terakhir = df.index[-1]
    sekarang_wita = datetime.now(pytz.timezone('Asia/Makassar'))
    
    confidence = max(latest_prob, 1 - latest_prob) * 100
    if latest_prob >= 0.60: arah, rekomendasi = "NAIK KUAT 🚀", "MOMENTUM EMAS UNTUK BELI (BUY)"
    elif latest_prob > 0.50: arah, rekomendasi = "Cenderung NAIK 📈", "POTENSI NAIK (BUY BERTAHAP)"
    elif latest_prob <= 0.40: arah, rekomendasi = "TURUN KUAT 🚨", "BAHAYA! PASAR ANJLOK (STRONG SELL)"
    else: arah, rekomendasi = "Cenderung TURUN 📉", "POTENSI TURUN, LEBIH BAIK MENUNGGU (WAIT/SELL)"

    if exposure == 0:
        saran_trading = "Lebih baik simpan aset/uang tunai dulu karena risiko penurunan sedang tinggi."
    elif latest_prob >= 0.60:
        saran_trading = "Momentum sangat bagus, pertimbangkan untuk masuk dengan alokasi modal maksimal yang disarankan."
    elif latest_prob > 0.50:
        saran_trading = "Pasar terlihat positif, pertimbangkan untuk masuk secara bertahap sesuai alokasi modal."
    else:
        saran_trading = "Tetap waspada, pantau pergerakan pasar sebelum mengambil keputusan besar."

    waktu_candle = waktu_data_terakhir.strftime('%H:%M WITA')
    selisih_menit = int((sekarang_wita - waktu_data_terakhir).total_seconds() / 60)
    status_sinkronisasi = f"⚡ Sangat Cepat (Data {selisih_menit}m lalu)" if selisih_menit <= 15 else f"⏳ Normal (Data {selisih_menit}m lalu)"
    
    vwap_status = "Aman (Harga di atas rata-rata tarikan bandar)" if latest_close_usd > df['VWAP_24'].iloc[-1] else "Bahaya (Harga di bawah rata-rata tarikan bandar)"
    tren_status = "Kuat (Pasar sedang aktif bergerak)" if df['ADX'].iloc[-1] > 25 else "Lemah / Sideways (Pasar ragu-ragu)"

    # --- FORMAT ASLI MILIK ANDA ---
    pesan = f"💎 *LAPORAN TRADING AI GODMODE (ANTI-REPAINT)* 💎\n"
    pesan += f"_{sekarang_wita.strftime('%d %B %Y | %H:%M WITA')}_\n\n"
    
    pesan += f"⚙️ *Kondisi Server & Sistem:*\n"
    pesan += f"├ Sinkronisasi: {status_sinkronisasi}\n"
    pesan += f"├ Waktu Candle: {waktu_candle}\n"
    pesan += f"└ Beban Engine AI: {waktu_eksekusi:.1f} detik\n\n"
    
    pesan += f"💰 *Harga BTC Sekarang:* {format_rupiah(indodax_live_idr)}\n\n"
    
    pesan += f"🤖 *PREDIKSI AI 6 JAM KE DEPAN:*\n"
    pesan += f"_(Otak utama yang menebak arah tren pasar)_\n"
    pesan += f"├ Arah Harga: *{arah}*\n"
    pesan += f"├ Keyakinan AI: *{confidence:.1f}%* (Makin tinggi makin akurat)\n"
    pesan += f"├ Akurasi AI: *{ml_accuracy:.1f}%* (Tanpa Data Leakage)\n"
    pesan += f"└ Cek 6 Jam Lalu: {eval_msg}\n\n"
    
    pesan += f"🔮 *SIMULASI HARGA 24 JAM (MONTE CARLO):*\n"
    pesan += f"_(AI mensimulasikan ribuan skenario untuk menebak harga besok)_\n"
    pesan += f"├ Harga Harapan: {format_rupiah(expected_24h_idr)} (Target rata-rata wajar)\n"
    pesan += f"└ Batas Apes (VaR 95%): {format_rupiah(var95_idr)}\n"
    pesan += f"   ↳ _Penjelasan: Sangat cocok dijadikan titik Stop Loss (SL)._\n\n"
    
    pesan += f"📊 *SARAN MANAJEMEN MODAL & PASAR:*\n"
    pesan += f"_(Panduan agar saldo Indodax tetap aman)_\n"
    pesan += f"├ Alokasi Modal Aman: Gunakan maksimal *{exposure}%* dari total saldomu.\n"
    pesan += f"├ Tren Saat Ini (ADX): {tren_status}\n"
    pesan += f"└ Posisi Bandar (VWAP): {vwap_status}\n\n"
    
    pesan += f"📰 *SENTIMEN BERITA DUNIA:* {news_status}\n\n"
    
    pesan += f"📌 *KESIMPULAN AKHIR:*\n"
    pesan += f"*{rekomendasi}*\n\n"
    
    pesan += f"💡 *Saran Trading:* {saran_trading}\n"
    pesan += f"🎯 _(Opsional) Pasang Cut Loss di {format_rupiah(var95_idr)} dan Take Profit di {format_rupiah(expected_24h_idr)}._\n\n"
    pesan += f"ℹ️ _Note: Khusus di foto Grafik menggunakan format Dolar (USD) agar bentuk grafiknya stabil._"

    chart_main = "chart_main.png"
    chart_zoom = "chart_zoom.png"
    
    plot_professional_analysis(df, next_time, next_target_usd, chart_main)
    plot_zoomed_analysis(df, next_time, next_target_usd, chart_zoom)
    
    send_to_telegram(pesan, [chart_main, chart_zoom])

if __name__ == "__main__":
    main()