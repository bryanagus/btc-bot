# ==============================================================================
# BTC QUANT GODMODE PRO MAX ENGINE (TIMEFRAME 1 JAM)
# Pembaruan: Indodax Live API, VADER NLP, Dynamic Kelly, No Data Leakage
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

warnings.filterwarnings('ignore')

# ================= KONFIGURASI TELEGRAM & SERVER =================
# Mengambil dari GitHub Secrets agar aman dan tidak terekspos public
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️ PERINGATAN: TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID belum di-set di Environment GitHub Secrets!")
# ==================================================================

def format_rupiah(angka):
    if pd.isna(angka): 
        return "Rp 0"
    return f"Rp {angka:,.0f}".replace(',', '.')

# ------------------------------------------------------------------------------
# 1. MODUL SENTIMEN BERITA DENGAN NLP (VADER)
# ------------------------------------------------------------------------------
def fetch_crypto_news_sentiment():
    print("[*] Mengumpulkan dan menganalisis berita Kripto global dengan NLP...")
    rss_urls = [
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://cointelegraph.com/rss',
        'https://cryptopotato.com/feed/'
    ]
    
    analyzer = SentimentIntensityAnalyzer()
    compound_scores = []
    unique_news = set()
    
    for url in rss_urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=5)
            root = ET.fromstring(response.read())
            
            for item in root.findall('.//item')[:10]:
                title = item.find('title').text
                if not title: 
                    continue
                
                clean_title = re.sub(r'[^\w\s]', '', title.lower())
                if clean_title not in unique_news:
                    unique_news.add(clean_title)
                    # Analisis VADER NLP
                    sentiment = analyzer.polarity_scores(title)
                    compound_scores.append(sentiment['compound'])
        except Exception as e:
            continue

    if not compound_scores:
        return "TIDAK ADA DATA BERITA ⚪"

    avg_score = sum(compound_scores) / len(compound_scores)
    
    if avg_score >= 0.25: 
        return f"SANGAT POSITIF 🚀 ({len(unique_news)} Berita)"
    elif avg_score > 0.05: 
        return f"POSITIF RINGAN 🟢 ({len(unique_news)} Berita)"
    elif avg_score <= -0.25: 
        return f"SANGAT NEGATIF 🚨 ({len(unique_news)} Berita)"
    elif avg_score < -0.05: 
        return f"NEGATIF RINGAN 🔴 ({len(unique_news)} Berita)"
    
    return f"NETRAL/SEIMBANG ⚪ ({len(unique_news)} Berita)"

# ------------------------------------------------------------------------------
# 2. MODUL DATA (YFINANCE + INDODAX LIVE API) & FEATURE ENGINEERING
# ------------------------------------------------------------------------------
def fetch_and_engineer_features(period='180d', interval='1h'):
    print("[*] Mengunduh data pasar historis...")
    df = yf.download('BTC-USD', period=period, interval=interval, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.droplevel(1)

    if df.index.tzinfo is None: 
        df.index = df.index.tz_localize('UTC')
    
    df.index = df.index.tz_convert('Asia/Makassar') # WITA

    # Konversi ke IDR
    try:
        idr_data = yf.download('IDR=X', period='5d', progress=False)
        if isinstance(idr_data.columns, pd.MultiIndex): 
            idr_data.columns = idr_data.columns.droplevel(1)
        kurs_idr = float(idr_data['Close'].iloc[-1])
    except:
        kurs_idr = 16000.0

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns: 
            df[col] = df[col] * kurs_idr

    # INJEKSI HARGA REAL-TIME DARI INDODAX
    print("[*] Mengambil harga live dari API Indodax...")
    try:
        indodax_req = requests.get('https://indodax.com/api/ticker/btcidr', timeout=5).json()
        live_price = float(indodax_req['ticker']['last'])
        # Timpa harga close terakhir dengan harga live Indodax agar AI menebak dari harga detik ini
        df.loc[df.index[-1], 'Close'] = live_price
    except Exception as e:
        print(f"[!] Gagal mengambil API Indodax, menggunakan harga konversi YFinance. Error: {e}")

    # --- INDIKATOR UNTUK CHART ---
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_20 * 2.0)
    df['BB_Lower'] = df['BB_Middle'] - (std_20 * 2.0)
    
    # RSI & StochRSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))
    df['StochRSI'] = (df['RSI'] - df['RSI'].rolling(14).min()) / (df['RSI'].rolling(14).max() - df['RSI'].rolling(14).min())

    # ATR & ADX
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1).rolling(14).mean()
    
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['ADX'] = (100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])).ewm(alpha=1/14, adjust=False).mean()

    # VWAP
    if 'Volume' in df.columns:
        df['VP'] = ((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']
        df['VWAP_24'] = df['VP'].rolling(window=24).sum() / df['Volume'].rolling(window=24).sum()
    else:
        df['VWAP_24'] = df['Close']

    # --- FITUR UNTUK MACHINE LEARNING ---
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA_Spread"] = (df["EMA20"] - df["EMA50"]) / df["Close"]
    
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    df["MACD_Hist"] = macd - macd.ewm(span=9).mean()
    
    df["Return_1H"] = df["Close"].pct_change()
    df["Return_3H"] = df["Close"].pct_change(3)
    df["Return_6H"] = df["Close"].pct_change(6)
    
    df["Volatility"] = (df["High"] - df["Low"]).rolling(14).mean() / df["Close"]
    
    if 'Volume' in df.columns:
        df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(24).mean() 
    else:
        df["Volume_Ratio"] = 1
    
    def calc_slope(x):
        try: return np.polyfit(range(len(x)), x, 1)[0]
        except: return 0
            
    df["Trend_Slope"] = df["Close"].rolling(12).apply(calc_slope, raw=True)
    df["Momentum_Accel"] = df["Return_1H"].diff()
    df["Regime"] = np.where(df["EMA20"] > df["EMA50"], 1, 0)
    
    # Target ML: 1 jika candle berikutnya naik, 0 jika turun
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    
    features_cols = ["EMA_Spread","RSI","MACD_Hist","Return_1H","Return_3H","Return_6H",
                     "Volatility","Volume_Ratio","Trend_Slope","Momentum_Accel","Regime"]
    
    return df, features_cols

# ------------------------------------------------------------------------------
# 3. MODUL ENSEMBLE MACHINE LEARNING (ANTI LEAKAGE)
# ------------------------------------------------------------------------------
def train_and_predict(df, features):
    print("[*] Melatih AI Ensemble (Tanpa Data Leakage)...")
    
    train_df = df.iloc[:-1].dropna(subset=features + ["Target"])
    X_train = train_df[features].values
    y_train = train_df["Target"].values
    latest_features = df[features].iloc[-1:].values

    lr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    # AKURASI DENGAN CROSS-VALIDATION YANG BENAR (Fit Scaler di dalam loop)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for train_index, test_index in tscv.split(X_train):
        X_tr, X_te = X_train[train_index], X_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        
        # Mencegah kebocoran data masa depan
        scaler_cv = StandardScaler()
        X_tr_scaled = scaler_cv.fit_transform(X_tr)
        X_te_scaled = scaler_cv.transform(X_te)
        
        lr.fit(X_tr_scaled, y_tr)
        rf.fit(X_tr_scaled, y_tr)
        gb.fit(X_tr_scaled, y_tr)
        
        prob_lr_cv = lr.predict_proba(X_te_scaled)[:, 1]
        prob_rf_cv = rf.predict_proba(X_te_scaled)[:, 1]
        prob_gb_cv = gb.predict_proba(X_te_scaled)[:, 1]
        
        ensemble_cv_prob = (prob_lr_cv + prob_rf_cv + prob_gb_cv) / 3
        fold_accuracy = accuracy_score(y_te, (ensemble_cv_prob > 0.5).astype(int))
        cv_scores.append(fold_accuracy)
        
    accuracy = np.mean(cv_scores) * 100
    
    # FINAL TRAINING untuk Prediksi Hari Ini
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_train)
    X_latest_scaled = final_scaler.transform(latest_features)

    lr_calibrated = CalibratedClassifierCV(LogisticRegression(), method="sigmoid", cv=3)
    lr_calibrated.fit(X_scaled, y_train)
    rf.fit(X_scaled, y_train)
    gb.fit(X_scaled, y_train)

    prob_lr_past = lr_calibrated.predict_proba(X_scaled)[:, 1]
    prob_rf_past = rf.predict_proba(X_scaled)[:, 1]
    prob_gb_past = gb.predict_proba(X_scaled)[:, 1]
    prob_all_past = (prob_lr_past + prob_rf_past + prob_gb_past) / 3

    prob_lr = lr_calibrated.predict_proba(X_latest_scaled)[0,1]
    prob_rf = rf.predict_proba(X_latest_scaled)[0,1]
    prob_gb = gb.predict_proba(X_latest_scaled)[0,1]
    latest_prob = (prob_lr + prob_rf + prob_gb) / 3
    
    # LOGIKA "MESIN WAKTU" (Untuk mengevaluasi prediksi 1 jam sebelumnya)
    past_train_df = df.iloc[:-2].dropna(subset=features + ["Target"])
    if len(past_train_df) > 0:
        scaler_past = StandardScaler()
        X_past_train = scaler_past.fit_transform(past_train_df[features].values)
        y_past_train = past_train_df["Target"].values
        X_past_target = scaler_past.transform(df[features].iloc[-2:-1].fillna(0).values) 

        lr_mesinwaktu = LogisticRegression()
        rf_mesinwaktu = RandomForestClassifier(n_estimators=100, random_state=42)
        gb_mesinwaktu = GradientBoostingClassifier(random_state=42)

        lr_mesinwaktu.fit(X_past_train, y_past_train)
        rf_mesinwaktu.fit(X_past_train, y_past_train)
        gb_mesinwaktu.fit(X_past_train, y_past_train)

        p_lr_past = lr_mesinwaktu.predict_proba(X_past_target)[0,1]
        p_rf_past = rf_mesinwaktu.predict_proba(X_past_target)[0,1]
        p_gb_past = gb_mesinwaktu.predict_proba(X_past_target)[0,1]
        true_past_prob = (p_lr_past + p_rf_past + p_gb_past) / 3
    else:
        true_past_prob = 0.5

    return latest_prob, accuracy, true_past_prob, prob_all_past, train_df.index

# ------------------------------------------------------------------------------
# 4. MODUL MONTE CARLO & DYNAMIC KELLY CRITERION
# ------------------------------------------------------------------------------
def monte_carlo_simulation(price, vol, steps=24, sims=2000):
    print(f"[*] Menjalankan {sims} Simulasi Monte Carlo untuk {steps} jam ke depan...")
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

def position_sizing_kelly(prob, current_price, expected_price, var95):
    # R = Reward / Risk
    reward = expected_price - current_price
    risk = current_price - var95
    
    # Hindari pembagian dengan nol atau R negatif
    if risk > 0 and reward > 0:
        r_ratio = reward / risk
    else:
        r_ratio = 1.0 # Default fallback
        
    edge = prob - ((1 - prob) / r_ratio)
    kelly = max(edge, 0)
    
    # Batasi maksimal 25% dari total portofolio agar manajemen risiko terjaga
    size = min(kelly, 0.25) 
    return round(size * 100, 2)

# ------------------------------------------------------------------------------
# 5A. VISUALISASI CHART UTAMA (80 JAM)
# ------------------------------------------------------------------------------
def plot_professional_analysis(df, filename="chart_main.png"):
    plot_data = df.tail(80) 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    ax1.plot(plot_data.index, plot_data['Close'], label='Harga Asli BTC', color='black', linewidth=2)
    ax1.plot(plot_data.index, plot_data['VWAP_24'], label='Garis Harga Bandar (VWAP)', color='#ff7f0e', linestyle='-.', linewidth=2)
    ax1.plot(plot_data.index, plot_data['MA_50'], label='Trend Menengah (MA50)', color='blue', alpha=0.6)
    ax1.fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], color='gray', alpha=0.15, label='Batas Harga Normal')

    if 'Garis_Prediksi' in plot_data.columns:
        ax1.plot(plot_data.index, plot_data['Garis_Prediksi'], label='Riwayat Prediksi AI', color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        last_time = plot_data.index[-1]
        next_time = last_time + pd.Timedelta(hours=1)
        next_pred = plot_data['AI_Target'].iloc[-1]
        
        ax1.scatter(next_time, next_pred, color='red', s=60, marker='o', zorder=10, label=f'Target AI Jam Depan: {format_rupiah(next_pred)}')
        ax1.plot([last_time, next_time], [plot_data['Garis_Prediksi'].iloc[-1], next_pred], color='red', linestyle='--', linewidth=2)

    ax1.set_title('GRAFIK UTAMA AI & JEJAK RIWAYAT PREDIKSI (80 JAM)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Harga (IDR)')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(',', '.')))
    
    myFmt = mdates.DateFormatter('%d %b\n%H:%M')
    hour_locator = mdates.HourLocator(interval=6) 
    
    ax2.plot(plot_data.index, plot_data['RSI'], color='purple', label='RSI (Momentum Harga)')
    ax2.plot(plot_data.index, plot_data['StochRSI']*100, color='cyan', alpha=0.5, label='StochRSI (Lebih Sensitif)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.fill_between(plot_data.index, plot_data['RSI'], 70, where=(plot_data['RSI'] >= 70), facecolor='red', alpha=0.3)
    ax2.fill_between(plot_data.index, plot_data['RSI'], 30, where=(plot_data['RSI'] <= 30), facecolor='green', alpha=0.3)
    ax2.set_ylabel('Momentum')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')

    ax3.plot(plot_data.index, plot_data['ADX'], color='brown', linewidth=2, label='ADX (Kekuatan Tren)')
    ax3.axhline(25, color='black', linestyle='--', alpha=0.8, label='Batas Tren Kuat (>25)')
    ax3.fill_between(plot_data.index, plot_data['ADX'], 25, where=(plot_data['ADX'] >= 25), facecolor='gold', alpha=0.4)
    ax3.set_ylabel('Kekuatan Tren')
    ax3.set_ylim(0, 60)
    ax3.legend(loc='upper left')

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(myFmt)
        ax.xaxis.set_major_locator(hour_locator) 
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 5B. VISUALISASI CHART ZOOM
# ------------------------------------------------------------------------------
def plot_zoomed_analysis(df, filename="chart_zoom.png"):
    plot_data = df.tail(8) 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(plot_data.index, plot_data['Close'], label='Harga Asli BTC', color='black', linewidth=3)
    ax1.plot(plot_data.index, plot_data['VWAP_24'], label='Garis Harga Bandar (VWAP)', color='#ff7f0e', linestyle='-.', linewidth=2)
    ax1.fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], color='gray', alpha=0.15)

    if 'Garis_Prediksi' in plot_data.columns:
        ax1.plot(plot_data.index, plot_data['Garis_Prediksi'], label='Riwayat Prediksi AI', color='red', linestyle='--', linewidth=2.5, alpha=0.8)
        last_time = plot_data.index[-1]
        next_time = last_time + pd.Timedelta(hours=1)
        next_pred = plot_data['AI_Target'].iloc[-1]
        
        ax1.scatter(next_time, next_pred, color='red', s=120, marker='o', zorder=10)
        ax1.plot([last_time, next_time], [plot_data['Garis_Prediksi'].iloc[-1], next_pred], color='red', linestyle='--', linewidth=2.5)

    ax1.set_title('🔍 ZOOM-IN 6 JAM TERAKHIR & PREDIKSI ARAH', fontsize=18, fontweight='bold', pad=15)
    ax1.set_ylabel('Harga (IDR)')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=11)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(',', '.')))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 6. MODUL TELEGRAM PENGIRIMAN
# ------------------------------------------------------------------------------
def send_to_telegram(message, image_paths):
    print("[*] Mengirim laporan teks dan grafik ke Telegram...")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Pengiriman dibatalkan. Token Telegram tidak disetel.")
        return
        
    url_message = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload_msg = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    requests.post(url_message, data=payload_msg)

    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    for path in image_paths:
        with open(path, 'rb') as photo:
            payload_photo = {'chat_id': TELEGRAM_CHAT_ID}
            requests.post(url_photo, data=payload_photo, files={'photo': photo})
    print("[*] Selesai Mengirim!")

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    print("==========================================================")
    print("      QUANT GODMODE PRO MAX ENGINE - BITCOIN INDODAX      ")
    print("==========================================================")
    
    # 1. Fetch & Proses Data
    df, features = fetch_and_engineer_features()
    news_status = fetch_crypto_news_sentiment()
    latest_close = df['Close'].iloc[-1]
    volatility = df["Volatility"].iloc[-1]
    
    # 2. AI Machine Learning
    latest_prob, ml_accuracy, past_prob, all_probs, train_idx = train_and_predict(df, features)
    
    df['AI_Prob'] = np.nan
    df.loc[train_idx, 'AI_Prob'] = all_probs
    df.iloc[-1, df.columns.get_loc('AI_Prob')] = latest_prob
    
    df['AI_Target'] = df['Close'] + (df['ATR'] * (df['AI_Prob'] - 0.5) * 2)
    df['Garis_Prediksi'] = df['AI_Target'].shift(1)
    
    # 3. Quant / Risk Management
    expected_24h, var95 = monte_carlo_simulation(latest_close, volatility)
    exposure = position_sizing_kelly(latest_prob, latest_close, expected_24h, var95)
    
    # 4. Evaluasi Performa
    waktu_data_terakhir = df.index[-1]
    sekarang_wita = datetime.now(pytz.timezone('Asia/Makassar'))
    selisih_menit = int(abs((sekarang_wita - waktu_data_terakhir).total_seconds()) / 60)
    
    if selisih_menit <= 20: info_server = f"⚡ Sangat Cepat (Data {selisih_menit}m lalu)"
    elif selisih_menit <= 60: info_server = f"✅ Normal (Data {selisih_menit}m lalu)"
    else: info_server = f"🐢 Delay Eksekusi (Data {selisih_menit}m lalu)"
    
    # Logika Evaluasi dengan Mesin Waktu AI yang Jujur
    prev_close = df['Close'].iloc[-2]
    pred_arah_lalu = "NAIK" if past_prob > 0.5 else "TURUN"
    arah_asli = "Naik" if latest_close > prev_close else "Turun"

    if (past_prob > 0.5 and latest_close > prev_close) or (past_prob <= 0.5 and latest_close <= prev_close): 
        eval_msg = f"BENAR ✅ (Sejam lalu AI prediksi {pred_arah_lalu}, dan terbukti {arah_asli})"
    else: 
        eval_msg = f"SALAH ❌ (Sejam lalu AI prediksi {pred_arah_lalu}, tapi harga malah {arah_asli})"

    confidence = max(latest_prob, 1 - latest_prob) * 100
    if latest_prob >= 0.60:
        arah, rekomendasi = "NAIK KUAT 🚀", "MOMENTUM EMAS UNTUK BELI (STRONG BUY)"
    elif latest_prob > 0.50:
        arah, rekomendasi = "Cenderung NAIK 📈", "POTENSI NAIK, BOLEH BELI JIKA TREN KUAT (BUY)"
    elif latest_prob <= 0.40:
        arah, rekomendasi = "TURUN KUAT 🚨", "BAHAYA! PASAR ANJLOK (STRONG SELL)"
    else:
        arah, rekomendasi = "Cenderung TURUN 📉", "POTENSI TURUN, LEBIH BAIK MENUNGGU (WAIT/SELL)"

    # 5. Susun Pesan Telegram
    pesan = f"💎 *LAPORAN TRADING AI GODMODE* 💎\n"
    pesan += f"_{sekarang_wita.strftime('%d %B %Y | %H:%M WITA')}_\n\n"
    
    pesan += f"💰 *Harga BTC Sekarang:* {format_rupiah(latest_close)}\n"
    pesan += f"⏱️ *Kondisi Server:* {info_server}\n\n"
    
    pesan += f"🤖 *PREDIKSI AI 1 JAM KE DEPAN:*\n"
    pesan += f"_(Otak utama yang menebak arah tren pasar)_\n"
    pesan += f"├ Arah Harga: *{arah}*\n"
    pesan += f"├ Keyakinan AI: *{confidence:.1f}%*\n"
    pesan += f"├ Akurasi AI: {ml_accuracy:.1f}% (Tanpa Data Leakage)\n"
    pesan += f"└ Cek Sejam Lalu: {eval_msg}\n\n"
    
    pesan += f"🔮 *SIMULASI HARGA 24 JAM (MONTE CARLO):*\n"
    pesan += f"├ Harga Harapan: {format_rupiah(expected_24h)} (Target rata-rata wajar)\n"
    pesan += f"└ Batas Apes (VaR 95%): {format_rupiah(var95)}\n"
    pesan += f"   _↳ Penjelasan: Sangat cocok dijadikan titik Stop Loss (SL)._\n\n"
    
    pesan += f"📊 *SARAN MANAJEMEN MODAL & PASAR:*\n"
    pesan += f"├ Alokasi Modal Aman: Gunakan *{exposure}%* dari total uangmu.\n"
    pesan += f"├ Tren Saat Ini (ADX): {'Kuat (Pasar sedang aktif)' if df['ADX'].iloc[-1] > 25 else 'Lemah (Sideways, jangan agresif)'}\n"
    pesan += f"└ Posisi Bandar (VWAP): {'Aman (Di atas rata-rata tarikan bandar)' if latest_close > df['VWAP_24'].iloc[-1] else 'Waspada (Di bawah rata-rata bandar)'}\n\n"
    
    pesan += f"📰 *SENTIMEN BERITA DUNIA:* {news_status}\n\n"
    
    pesan += f"📌 *KESIMPULAN AKHIR:*\n"
    pesan += f"*{rekomendasi}*\n\n"
    
    if latest_prob > 0.5:
        pesan += f"💡 _Saran Trading: Pasang Cut Loss di dekat {format_rupiah(var95)} dan Jual Untung (TP) di kisaran {format_rupiah(expected_24h)}._\n"
    else:
        pesan += f"💡 _Saran Trading: Lebih baik simpan aset/uang tunai dulu karena risiko penurunan sedang tinggi._\n"
        
    pesan += "\n_ℹ️ Disclaimer: Angka ini dihitung oleh algoritma AI. Gunakan sebagai alat bantu, bukan jaminan pasti 100%._"

    # 6. Eksekusi Chart & Kirim
    chart_main = "chart_main.png"
    chart_zoom = "chart_zoom.png"
    
    plot_professional_analysis(df, chart_main)
    plot_zoomed_analysis(df, chart_zoom)
    
    send_to_telegram(pesan, [chart_main, chart_zoom])

if __name__ == "__main__":
    main()
