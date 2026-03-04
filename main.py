# ==============================================================================
# BTC QUANT GODMODE PRO MAX ENGINE (TIMEFRAME 1 JAM)
# Fitur: Ensemble Machine Learning, Monte Carlo, Kelly Risk Engine, 
#        Multi-Source News, ADX, VWAP, Telegram Reporting & Charting
# Pembaruan: Laporan Ramah Awam & Visual Garis Prediksi Riwayat AI
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
    print("⚠️ PERINGATAN: TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID belum di-set di Environment!")
# ==================================================================

def format_rupiah(angka):
    if pd.isna(angka): return "Rp 0"
    return f"Rp {angka:,.0f}".replace(',', '.')

# ------------------------------------------------------------------------------
# 1. MODUL SENTIMEN BERITA (PRO MAX)
# ------------------------------------------------------------------------------
def fetch_crypto_news_sentiment():
    print("[*] Mengumpulkan dan menganalisis berita Kripto global...")
    rss_urls = [
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://cointelegraph.com/rss',
        'https://cryptopotato.com/feed/'
    ]
    bullish_keywords = ['surge', 'jump', 'rise', 'bull', 'high', 'adopt', 'approve', 'gain', 'positive', 'buy', 'up', 'soar', 'breakout', 'record']
    bearish_keywords = ['drop', 'fall', 'crash', 'bear', 'low', 'ban', 'reject', 'lose', 'negative', 'sell', 'down', 'hack', 'scam', 'plunge']
    
    bullish_score, bearish_score = 0, 0
    unique_news = set()
    
    for url in rss_urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=5)
            root = ET.fromstring(response.read())
            
            for item in root.findall('.//item')[:10]:
                title = item.find('title').text
                if not title: continue
                clean_title = re.sub(r'[^\w\s]', '', title.lower())
                
                if clean_title not in unique_news:
                    unique_news.add(clean_title)
                    for word in bullish_keywords:
                        if word in clean_title: bullish_score += 1
                    for word in bearish_keywords:
                        if word in clean_title: bearish_score += 1
        except Exception:
            continue

    selisih = bullish_score - bearish_score
    if selisih >= 3: 
        return f"SANGAT POSITIF 🚀 ({len(unique_news)} Berita)"
    elif selisih > 0: 
        return f"POSITIF RINGAN 🟢 ({len(unique_news)} Berita)"
    elif selisih <= -3: 
        return f"SANGAT NEGATIF 🚨 ({len(unique_news)} Berita)"
    elif selisih < 0: 
        return f"NEGATIF RINGAN 🔴 ({len(unique_news)} Berita)"
    
    return f"NETRAL/SEIMBANG ⚪ ({len(unique_news)} Berita)"

# ------------------------------------------------------------------------------
# 2. MODUL DATA & FEATURE ENGINEERING (GABUNGAN)
# ------------------------------------------------------------------------------
def fetch_and_engineer_features(period='180d', interval='1h'):
    print("[*] Mengunduh data pasar dan memproses Feature Engineering...")
    df = yf.download('BTC-USD', period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)

    if df.index.tzinfo is None: df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Asia/Makassar') # WITA

    try:
        idr_data = yf.download('IDR=X', period='5d', progress=False)
        if isinstance(idr_data.columns, pd.MultiIndex): idr_data.columns = idr_data.columns.droplevel(1)
        kurs_idr = float(idr_data['Close'].iloc[-1])
    except:
        kurs_idr = 16000.0

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns: df[col] = df[col] * kurs_idr

    # --- INDIKATOR UNTUK CHART (PRO MAX) ---
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

    # --- FITUR UNTUK MACHINE LEARNING (GODMODE) ---
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA_Spread"] = (df["EMA20"] - df["EMA50"]) / df["Close"]
    
    ema12, ema26 = df["Close"].ewm(span=12).mean(), df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    df["MACD_Hist"] = macd - macd.ewm(span=9).mean()
    
    df["Return_1H"] = df["Close"].pct_change()
    df["Return_3H"] = df["Close"].pct_change(3)
    df["Return_6H"] = df["Close"].pct_change(6)
    
    df["Volatility"] = (df["High"] - df["Low"]).rolling(14).mean() / df["Close"]
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(24).mean() if 'Volume' in df.columns else 1
    
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
# 3. MODUL ENSEMBLE MACHINE LEARNING & CALIBRATION (AKURASI JUJUR)
# ------------------------------------------------------------------------------
def train_and_predict(df, features):
    print("[*] Melatih AI Ensemble (RandomForest, GradientBoosting, LogisticReg)...")
    
    train_df = df.iloc[:-1].dropna(subset=features + ["Target"])
    X_train = train_df[features].values
    y_train = train_df["Target"].values
    latest_features = df[features].iloc[-1:]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_latest_scaled = scaler.transform(latest_features)

    lr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    # Validasi Silang (Cross Validation) agar hasil akurasi realistis, bukan 100% bohong
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_index, test_index in tscv.split(X_scaled):
        X_tr, X_te = X_scaled[train_index], X_scaled[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        
        lr.fit(X_tr, y_tr); rf.fit(X_tr, y_tr); gb.fit(X_tr, y_tr)
        
        ensemble_cv_prob = (lr.predict_proba(X_te)[:, 1] + rf.predict_proba(X_te)[:, 1] + gb.predict_proba(X_te)[:, 1]) / 3
        cv_scores.append(accuracy_score(y_te, (ensemble_cv_prob > 0.5).astype(int)))
        
    accuracy = np.mean(cv_scores) * 100
    
    # Kalibrasi dan Fit final ke seluruh data untuk menebak harga hari ini
    lr_calibrated = CalibratedClassifierCV(LogisticRegression(), method="sigmoid", cv=3)
    lr_calibrated.fit(X_scaled, y_train)
    rf.fit(X_scaled, y_train)
    gb.fit(X_scaled, y_train)

    # Dapatkan seluruh riwayat prediksi probabilitas dari masa lalu (Untuk ditarik garisnya di Chart)
    prob_all_past = (lr_calibrated.predict_proba(X_scaled)[:,1] + rf.predict_proba(X_scaled)[:,1] + gb.predict_proba(X_scaled)[:,1]) / 3

    # Prediksi saat ini
    prob_lr = lr_calibrated.predict_proba(X_latest_scaled)[0,1]
    prob_rf = rf.predict_proba(X_latest_scaled)[0,1]
    prob_gb = gb.predict_proba(X_latest_scaled)[0,1]
    
    latest_prob = (prob_lr + prob_rf + prob_gb) / 3
    past_prob = prob_all_past[-1]

    return latest_prob, accuracy, past_prob, prob_all_past, train_df.index

# ------------------------------------------------------------------------------
# 4. MODUL MONTE CARLO & MANAJEMEN RISIKO (GODMODE)
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
    var95 = np.percentile(final_prices, 5) # Harga terburuk dengan probabilitas 5%
    return expected, var95

def position_sizing_kelly(prob, volatility):
    edge = prob - (1 - prob)
    kelly = max(edge, 0)
    vol_adjust = min(1 / (volatility * 100), 1)
    size = kelly * vol_adjust
    size = min(size, 0.25) 
    return round(size * 100, 2)

# ------------------------------------------------------------------------------
# 5. MODUL VISUALISASI CHART (GRAFIK GARIS RIWAYAT AI)
# ------------------------------------------------------------------------------
def plot_professional_analysis(df, filename="chart.png"):
    plot_data = df.tail(80) 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])

    # Plot Utama
    ax1.plot(plot_data.index, plot_data['Close'], label='Harga Asli BTC', color='black', linewidth=2)
    ax1.plot(plot_data.index, plot_data['VWAP_24'], label='Garis Harga Bandar (VWAP)', color='#ff7f0e', linestyle='-.', linewidth=2)
    ax1.fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], color='gray', alpha=0.15, label='Batas Harga Normal (Bollinger)')

    # --- FITUR BARU: GARIS PREDIKSI AI ---
    if 'Garis_Prediksi' in plot_data.columns:
        ax1.plot(plot_data.index, plot_data['Garis_Prediksi'], label='Riwayat Prediksi AI (Masa Lalu)', color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Titik Prediksi Masa Depan (Jam Berikutnya)
        next_time = plot_data.index[-1] + pd.Timedelta(hours=1)
        next_pred = plot_data['AI_Target'].iloc[-1]
        ax1.scatter(next_time, next_pred, color='red', s=250, marker='*', zorder=10, label=f'Target AI Jam Depan: {format_rupiah(next_pred)}')
        ax1.plot([plot_data.index[-1], next_time], [plot_data['Garis_Prediksi'].iloc[-1], next_pred], color='red', linestyle=':', linewidth=2)

    ax1.set_title('GRAFIK AI & JEJAK RIWAYAT PREDIKSI - WITA', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Harga (IDR)')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(',', '.')))
    
    myFmt = mdates.DateFormatter('%d %b\n%H:%M')
    
    # RSI
    ax2.plot(plot_data.index, plot_data['RSI'], color='purple', label='RSI (Momentum Jual/Beli)')
    ax2.plot(plot_data.index, plot_data['StochRSI']*100, color='cyan', alpha=0.5, label='StochRSI (Lebih Sensitif)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Jenuh Beli (Rentan Turun)')
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5, label='Jenuh Jual (Rentan Naik)')
    ax2.fill_between(plot_data.index, plot_data['RSI'], 70, where=(plot_data['RSI'] >= 70), facecolor='red', alpha=0.3)
    ax2.fill_between(plot_data.index, plot_data['RSI'], 30, where=(plot_data['RSI'] <= 30), facecolor='green', alpha=0.3)
    ax2.set_ylabel('Momentum')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')

    # ADX
    ax3.plot(plot_data.index, plot_data['ADX'], color='brown', linewidth=2, label='ADX (Kekuatan Tren Harga)')
    ax3.axhline(25, color='black', linestyle='--', alpha=0.8, label='Batas Tren Kuat (>25)')
    ax3.fill_between(plot_data.index, plot_data['ADX'], 25, where=(plot_data['ADX'] >= 25), facecolor='gold', alpha=0.4)
    ax3.set_ylabel('Kekuatan Tren')
    ax3.set_ylim(0, 60)
    ax3.legend(loc='upper left')

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(myFmt)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 6. MODUL TELEGRAM PENGIRIMAN
# ------------------------------------------------------------------------------
def send_to_telegram(message, image_path):
    print("[*] Mengirim laporan Godmode Pro Max ke Telegram...")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'})
    with open(image_path, 'rb') as photo:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': photo})
    print("[*] Selesai!")

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    start_time = time.time()
    df, features = fetch_and_engineer_features()
    news_status = fetch_crypto_news_sentiment()
    latest_close = df['Close'].iloc[-1]
    volatility = df["Volatility"].iloc[-1]
    
    # AI ML & Inject Probabilitas untuk Membuat Garis
    latest_prob, ml_accuracy, past_prob, all_probs, train_idx = train_and_predict(df, features)
    
    df['AI_Prob'] = np.nan
    df.loc[train_idx, 'AI_Prob'] = all_probs
    df.iloc[-1, df.columns.get_loc('AI_Prob')] = latest_prob
    
    # Hitung Target AI berdasarkan probabilitas (Membentuk garis searah tren)
    df['AI_Target'] = df['Close'] + (df['ATR'] * (df['AI_Prob'] - 0.5) * 2)
    # Geser 1 jam ke depan untuk melihat apakah prediksi sejam lalu sesuai dengan harga asli sejam ini
    df['Garis_Prediksi'] = df['AI_Target'].shift(1)
    
    # Quant / Risk Management
    expected_24h, var95 = monte_carlo_simulation(latest_close, volatility)
    exposure = position_sizing_kelly(latest_prob, volatility)
    
    # 4. Evaluasi & Perhitungan Data
    waktu_eksekusi = time.time() - start_time
    waktu_data_terakhir = df.index[-1]
    sekarang_wita = datetime.now(pytz.timezone('Asia/Makassar'))
    selisih_menit = int(abs((sekarang_wita - waktu_data_terakhir).total_seconds()) / 60)
    
    if selisih_menit <= 20: info_server = f"⚡ Sangat Cepat (Data {selisih_menit}m lalu)"
    elif selisih_menit <= 60: info_server = f"✅ Normal (Data {selisih_menit}m lalu)"
    else: info_server = f"🐢 Delay Eksekusi (Data {selisih_menit}m lalu)"
    
    # Logika Bahasa Manusia untuk Evaluasi Jam Lalu
    prev_close = df['Close'].iloc[-2]
    if past_prob > 0.5 and latest_close > prev_close: 
        eval_msg = "BENAR ✅ (AI Prediksi NAIK, dan Harga Terbukti Naik)"
    elif past_prob <= 0.5 and latest_close <= prev_close: 
        eval_msg = "BENAR ✅ (AI Prediksi TURUN, dan Harga Terbukti Turun)"
    elif past_prob > 0.5 and latest_close <= prev_close: 
        eval_msg = "SALAH ❌ (AI Prediksi NAIK, tapi Harga Malah Turun)"
    else: 
        eval_msg = "SALAH ❌ (AI Prediksi TURUN, tapi Harga Malah Naik)"

    confidence = max(latest_prob, 1 - latest_prob) * 100
    if latest_prob >= 0.60:
        arah, rekomendasi = "NAIK KUAT 🚀", "MOMENTUM EMAS UNTUK BELI (STRONG BUY)"
    elif latest_prob > 0.50:
        arah, rekomendasi = "Cenderung NAIK 📈", "POTENSI NAIK, BOLEH BELI JIKA TREN KUAT (BUY)"
    elif latest_prob <= 0.40:
        arah, rekomendasi = "TURUN KUAT 🚨", "BAHAYA! PASAR ANJLOK (STRONG SELL)"
    else:
        arah, rekomendasi = "Cenderung TURUN 📉", "POTENSI TURUN, LEBIH BAIK MENUNGGU (WAIT/SELL)"

    # ==========================================================
    # PENYUSUNAN PESAN TELEGRAM (BAHASA AWAM & INFORMATIF)
    # ==========================================================
    pesan = f"💎 *LAPORAN TRADING AI GODMODE* 💎\n"
    pesan += f"_{sekarang_wita.strftime('%d %B %Y | %H:%M WITA')}_\n\n"
    
    pesan += f"💰 *Harga BTC Sekarang:* {format_rupiah(latest_close)}\n"
    pesan += f"⏱️ *Kondisi Server:* {info_server}\n\n"
    
    pesan += f"🤖 *PREDIKSI AI 1 JAM KE DEPAN:*\n"
    pesan += f"_(Otak utama yang menebak arah tren pasar)_\n"
    pesan += f"├ Arah Harga: *{arah}*\n"
    pesan += f"├ Keyakinan AI: *{confidence:.1f}%* (Makin tinggi makin yakin)\n"
    pesan += f"├ Akurasi AI: {ml_accuracy:.1f}% (Berdasarkan ujian data riwayat)\n"
    pesan += f"└ Cek Sejam Lalu: {eval_msg}\n\n"
    
    pesan += f"🔮 *SIMULASI HARGA 24 JAM (MONTE CARLO):*\n"
    pesan += f"_(AI mensimulasikan ribuan skenario untuk menebak harga besok)_\n"
    pesan += f"├ Harga Harapan: {format_rupiah(expected_24h)} (Target rata-rata wajar)\n"
    pesan += f"└ Batas Apes (VaR 95%): {format_rupiah(var95)}\n"
    pesan += f"   _↳ Penjelasan: AI yakin 95% harga tidak akan jatuh lebih dalam dari angka ini. Jika kamu trading, angka ini sangat cocok dijadikan titik Stop Loss (SL)._\n\n"
    
    pesan += f"📊 *SARAN MANAJEMEN MODAL & PASAR:*\n"
    pesan += f"_(Panduan agar saldo Indodax tetap aman)_\n"
    pesan += f"├ Alokasi Modal Aman: Gunakan *{exposure}%* dari total uangmu.\n"
    pesan += f"├ Tren Saat Ini (ADX): {'Kuat (Pasar sedang aktif bergerak)' if df['ADX'].iloc[-1] > 25 else 'Lemah (Pasar sedang mendatar/sideways, jangan agresif)'}\n"
    pesan += f"└ Posisi Bandar (VWAP): {'Aman (Harga di atas rata-rata tarikan bandar)' if latest_close > df['VWAP_24'].iloc[-1] else 'Waspada (Harga di bawah rata-rata bandar)'}\n\n"
    
    pesan += f"📰 *SENTIMEN BERITA DUNIA:* {news_status}\n\n"
    
    pesan += f"📌 *KESIMPULAN AKHIR:*\n"
    pesan += f"*{rekomendasi}*\n"
    
    if latest_prob > 0.5:
        sl = var95  # Menggunakan VaR sebagai SL karena lebih akurat
        tp = expected_24h # Menggunakan Target Rata-rata sebagai TP
        pesan += f"💡 _Saran Trading: Pasang Cut Loss di dekat {format_rupiah(sl)} dan Jual Untung (TP) di kisaran {format_rupiah(tp)}._\n"
        
    pesan += "\n_ℹ️ Disclaimer: Angka ini dihitung oleh algoritma AI. Gunakan sebagai alat bantu, bukan jaminan pasti 100%._"

    # Eksekusi Chart & Kirim
    chart_filename = "godmode_chart.png"
    plot_professional_analysis(df, chart_filename)
    send_to_telegram(pesan, chart_filename)

if __name__ == "__main__":
    main()
