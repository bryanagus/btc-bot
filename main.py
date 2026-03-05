# ==============================================================================
# BTC QUANT GODMODE PRO MAX - DUAL ENGINE VERSION (1H & 6H)
# Fitur: Fast Backoff Retry, Auto-Risk, System Diagnostics, Zero Data Leakage
# Visual: Dual Target, Anti-Repainting, USD Stabil
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

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 

# ================= KONFIGURASI =================
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
HISTORY_FILE = "ai_history_log_v2.csv" # Ganti nama agar tidak bentrok dengan versi lama
# ===============================================

# Global Diagnostics Dictionary
diagnostics = {
    "api": "✅ Normal (API Terhubung)",
    "csv": "✅ Sinkron (Riwayat Terbaca)",
    "ai_1h": "✅ Optimal",
    "ai_6h": "✅ Optimal"
}

def format_rupiah(angka):
    if pd.isna(angka): return "Rp 0"
    return f"Rp {angka:,.0f}".replace(',', '.')

def format_usd(angka):
    if pd.isna(angka): return "$0"
    return f"${angka:,.0f}"

# ------------------------------------------------------------------------------
# 1. MODUL FETCH DATA DENGAN FAST BACKOFF RETRY (Max ~2 Menit)
# ------------------------------------------------------------------------------
def fetch_data_with_retry(period='90d', interval='1h'):
    print("[*] Mencoba menarik data dengan Fast Backoff...")
    delays = [5, 15, 30, 60] # Strategi mundur teratur
    
    for attempt, delay in enumerate(delays + [0]):
        try:
            # Tarik Global Data
            df = yf.download('BTC-USD', period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.droplevel(1)
            if df.index.tzinfo is None: 
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Makassar')
            
            # Tarik Kurs IDR
            idr_data = yf.download('IDR=X', period='5d', progress=False)
            if isinstance(idr_data.columns, pd.MultiIndex): 
                idr_data.columns = idr_data.columns.droplevel(1)
            kurs_idr = float(idr_data['Close'].dropna().iloc[-1])
            
            # Tarik API Indodax
            indodax_req = requests.get('https://indodax.com/api/ticker/btcidr', timeout=10).json()
            indodax_live_idr = float(indodax_req['ticker']['last'])
            
            return df, kurs_idr, indodax_live_idr
            
        except Exception as e:
            if attempt == len(delays):
                diagnostics["api"] = "❌ Gagal Terhubung (Server API Down)"
                raise Exception(f"API Timeout setelah semua percobaan gagal: {e}")
            print(f"[!] API Gagal. Menunggu {delay} detik sebelum coba lagi...")
            time.sleep(delay)

def fetch_crypto_news_sentiment():
    try:
        rss_urls = ['https://www.coindesk.com/arc/outboundfeeds/rss/', 'https://cointelegraph.com/rss']
        analyzer = SentimentIntensityAnalyzer()
        crypto_lexicon = {"bullish": 2.5, "bearish": -2.5, "rekt": -3.0, "moon": 2.5, "pump": 2.0, "dump": -2.5}
        analyzer.lexicon.update(crypto_lexicon)
        
        compound_scores = []
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0'}
        for url in rss_urls:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall('.//item')[:10]:
                    title = item.find('title').text
                    if title:
                        compound_scores.append(analyzer.polarity_scores(title)['compound'])
                        
        if not compound_scores: return "NETRAL ⚪"
        avg = sum(compound_scores) / len(compound_scores)
        if avg >= 0.25: return "SANGAT POSITIF 🚀"
        elif avg <= -0.25: return "SANGAT NEGATIF 🚨"
        return "NETRAL/SEIMBANG ⚪"
    except:
        return "BERITA TIDAK TERSEDIA ⚪"

# ------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (DUAL TARGET)
# ------------------------------------------------------------------------------
def engineer_features(df):
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_20 * 2.0)
    df['BB_Lower'] = df['BB_Middle'] - (std_20 * 2.0)
    
    delta = df['Close'].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1).rolling(14).mean()
    
    if 'Volume' in df.columns:
        df['VWAP_24'] = (((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']).rolling(24).sum() / df['Volume'].rolling(24).sum()
    else:
        df['VWAP_24'] = df['Close']

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA_Spread"] = (df["EMA20"] - df["EMA50"]) / df["Close"]
    
    df["Return_1H"] = df["Close"].pct_change()
    df["Volatility"] = df["Return_1H"].rolling(24).std()
    
    # DUAL TARGET (Kunci Analisis Multi-Timeframe)
    df["Target_1H"] = (df["Close"].shift(-1) > df["Close"]).astype(float)
    df["Target_6H"] = (df["Close"].shift(-6) > df["Close"]).astype(float)
    
    df = df.iloc[:-1] # Buang row terakhir yang blm selesai
    features_cols = ["EMA_Spread","RSI","Return_1H","Volatility"]
    return df, features_cols

# ------------------------------------------------------------------------------
# 3. MODUL AI JUJUR (DAPAT DIPAKAI UNTUK 1H & 6H)
# ------------------------------------------------------------------------------
def train_honest_model(df, features, target_col, shift_len):
    # Potong data sesuai target agar TIDAK NGINTIP masa depan
    train_df = df.iloc[:-shift_len].dropna(subset=features + [target_col])
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    
    X_live = df[features].iloc[-1:].fillna(0).values

    # Latih model
    lr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_live_scaled = scaler.transform(X_live)

    lr_cal = CalibratedClassifierCV(LogisticRegression(), method="sigmoid", cv=3)
    lr_cal.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)

    prob_lr = lr_cal.predict_proba(X_live_scaled)[0,1]
    prob_rf = rf.predict_proba(X_live_scaled)[0,1]
    prob_gb = gb.predict_proba(X_live_scaled)[0,1]
    
    latest_prob = (prob_lr + prob_rf + prob_gb) / 3
    
    # Hitung akurasi simple di data training
    p_rf_train = rf.predict(X_train_scaled)
    acc = accuracy_score(y_train, p_rf_train) * 100
    return latest_prob, acc

# ------------------------------------------------------------------------------
# 4. DATABASE CSV & EVALUASI AUTO-RISK
# ------------------------------------------------------------------------------
def manage_history_and_evaluate(df, prob_1h, prob_6h):
    current_time = df.index[-1]
    current_price = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    # Hitung Harga Prediksi Visual
    pred_1h_price = current_price + (atr * (prob_1h - 0.5) * 2)
    pred_6h_price = current_price + (atr * (prob_6h - 0.5) * 4)
    
    target_time_1h = current_time + pd.Timedelta(hours=1)
    target_time_6h = current_time + pd.Timedelta(hours=6)
    
    eval_1h_msg = "Menunggu data..."
    eval_6h_msg = "Menunggu data..."
    risk_multiplier = 1.0 # Default normal
    
    file_exists = os.path.isfile(HISTORY_FILE)
    
    try:
        if file_exists:
            history = pd.read_csv(HISTORY_FILE)
            history['timestamp_target_1h'] = pd.to_datetime(history['timestamp_target_1h'], utc=True).dt.tz_convert('Asia/Makassar')
            history['timestamp_target_6h'] = pd.to_datetime(history['timestamp_target_6h'], utc=True).dt.tz_convert('Asia/Makassar')
            
            # --- EVALUASI 1 JAM LALU ---
            row_1h = history[history['timestamp_target_1h'] == current_time]
            if not row_1h.empty and len(df) > 2:
                past_prob = row_1h['prob_1h'].values[0]
                prev_close = df['Close'].iloc[-2]
                arah_pred = "NAIK" if past_prob > 0.5 else "TURUN"
                arah_asli = "Naik" if current_price > prev_close else "Turun"
                if (past_prob > 0.5 and current_price > prev_close) or (past_prob <= 0.5 and current_price <= prev_close):
                    eval_1h_msg = f"BENAR ✅ (Nebak {arah_pred}, Asli {arah_asli})"
                else:
                    eval_1h_msg = f"SALAH ❌ (Nebak {arah_pred}, Asli {arah_asli})"

            # --- EVALUASI 6 JAM LALU ---
            row_6h = history[history['timestamp_target_6h'] == current_time]
            if not row_6h.empty and len(df) > 7:
                past_prob = row_6h['prob_6h'].values[0]
                prev_close = df['Close'].iloc[-7]
                arah_pred = "NAIK" if past_prob > 0.5 else "TURUN"
                arah_asli = "Naik" if current_price > prev_close else "Turun"
                if (past_prob > 0.5 and current_price > prev_close) or (past_prob <= 0.5 and current_price <= prev_close):
                    eval_6h_msg = f"BENAR ✅ (Nebak {arah_pred})"
                else:
                    eval_6h_msg = f"SALAH ❌ (Nebak {arah_pred})"

            # --- AUTO RISK CALCULATION (Hitung Win Rate 5 jam terakhir) ---
            recent_1h = history.tail(5)
            benar_count = 0
            total_eval = 0
            for _, row in recent_1h.iterrows():
                ts = row['timestamp_target_1h']
                if ts in df.index:
                    idx = df.index.get_loc(ts)
                    if idx > 0:
                        p_close = df['Close'].iloc[idx-1]
                        a_close = df['Close'].iloc[idx]
                        p_prob = row['prob_1h']
                        if (p_prob > 0.5 and a_close > p_close) or (p_prob <= 0.5 and a_close <= p_close):
                            benar_count += 1
                        total_eval += 1
            
            if total_eval >= 3:
                win_rate = benar_count / total_eval
                if win_rate < 0.4: risk_multiplier = 0.1     # Hancur, rem 90%
                elif win_rate < 0.6: risk_multiplier = 0.5   # Biasa, rem 50%
                else: risk_multiplier = 1.0                  # Bagus, gas full
                
    except Exception as e:
        diagnostics["csv"] = f"❌ Error Baca CSV ({str(e)[:20]})"
        risk_multiplier = 1.0

    # Simpan Data Baru
    try:
        with open(HISTORY_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['timestamp_target_1h', 'pred_1h_price', 'prob_1h', 'timestamp_target_6h', 'pred_6h_price', 'prob_6h'])
            writer.writerow([target_time_1h.isoformat(), pred_1h_price, prob_1h, target_time_6h.isoformat(), pred_6h_price, prob_6h])
    except:
        diagnostics["csv"] = "❌ Gagal Menulis CSV"

    return target_time_1h, pred_1h_price, eval_1h_msg, target_time_6h, pred_6h_price, eval_6h_msg, risk_multiplier

# ------------------------------------------------------------------------------
# 5. RISK MANAGEMENT (KELLY + MONTE CARLO)
# ------------------------------------------------------------------------------
def monte_carlo_simulation(price, vol, steps=1, sims=2000):
    paths = []
    for _ in range(sims):
        prices = [price]
        for _ in range(steps):
            shock = np.random.normal(0, vol)
            prices.append(prices[-1] * (1 + shock))
        paths.append(prices)
    final_prices = np.array(paths)[:, -1]
    return np.mean(final_prices), np.percentile(final_prices, 5)

def position_sizing_kelly(prob_1h, prob_6h, atr, risk_multiplier):
    # Kombinasikan keyakinan untuk Kelly dasar
    avg_prob = (prob_1h + prob_6h) / 2
    if avg_prob < 0.5: return 0.0
    
    reward, risk = atr * 3.5, atr * 1.5
    if risk <= 0: return 0.0 
    
    edge = avg_prob - ((1 - avg_prob) / (reward/risk))
    kelly = max(edge, 0)
    
    # TERAPKAN REM OTOMATIS
    size = min(kelly, 0.25) * risk_multiplier 
    return round(size * 100, 2)

# ------------------------------------------------------------------------------
# 6. VISUALISASI DUAL TARGET
# ------------------------------------------------------------------------------
def plot_professional_analysis(df, t_1h, p_1h, t_6h, p_6h, filename="chart_main.png"):
    plot_data = df.tail(80) 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 8))
    
    plt.plot(plot_data.index, plot_data['Close'], label='Harga Asli BTC', color='black', linewidth=2, zorder=5)
    plt.plot(plot_data.index, plot_data['VWAP_24'], label='Garis Bandar', color='#ff7f0e', linestyle='-.')
    
    last_time = plot_data.index[-1]
    last_price = plot_data['Close'].iloc[-1]
    
    # Plot Target 1H
    plt.scatter(t_1h, p_1h, color='blue', s=80, marker='o', zorder=10, label=f'Target 1 Jam')
    plt.plot([last_time, t_1h], [last_price, p_1h], color='blue', linestyle='--', linewidth=2)
    
    # Plot Target 6H
    plt.scatter(t_6h, p_6h, color='red', s=100, marker='X', zorder=10, label=f'Target 6 Jam')
    plt.plot([last_time, t_6h], [last_price, p_6h], color='red', linestyle='--', linewidth=2)

    plt.title('GRAFIK UTAMA AI DUAL ENGINE (80 JAM)', fontsize=16, fontweight='bold')
    plt.ylabel('Harga (USD)')
    plt.legend(loc='upper left', framealpha=0.9)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_zoomed_analysis(df, t_1h, p_1h, t_6h, p_6h, filename="chart_zoom.png"):
    plot_data = df.tail(12) 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(plot_data.index, plot_data['Close'], color='black', linewidth=3)
    ax1.fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], color='gray', alpha=0.15)

    last_time = plot_data.index[-1]
    last_price = plot_data['Close'].iloc[-1]
    
    ax1.scatter(t_1h, p_1h, color='blue', s=100)
    ax1.plot([last_time, t_1h], [last_price, p_1h], color='blue', linestyle='--')
    ax1.scatter(t_6h, p_6h, color='red', s=120, marker='X')
    ax1.plot([last_time, t_6h], [last_price, p_6h], color='red', linestyle='--')

    ax1.annotate(f"Harga Asli: {format_usd(last_price)}", (last_time, last_price), xytext=(0, -25), textcoords='offset points', ha='center', bbox=dict(boxstyle="round", fc="black", alpha=0.8), color="white")
    ax1.annotate(f"1H: {format_usd(p_1h)}", (t_1h, p_1h), xytext=(0, 15), textcoords='offset points', ha='center', bbox=dict(boxstyle="round", fc="blue", alpha=0.8), color="white")
    ax1.annotate(f"6H: {format_usd(p_6h)}", (t_6h, p_6h), xytext=(0, 15), textcoords='offset points', ha='center', bbox=dict(boxstyle="round", fc="red", alpha=0.8), color="white")

    ax1.set_title('🔍 ZOOM PITA BOLLINGER & TARGET GANDA', fontsize=16, fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ------------------------------------------------------------------------------
# 7. TELEGRAM SENDER (3 TAHAP)
# ------------------------------------------------------------------------------
def send_telegram_messages(pesan_utama, chart_paths, pesan_diag):
    if not TELEGRAM_BOT_TOKEN: return
    
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    # Kirim Pesan 1 (Utama)
    requests.post(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_utama, 'parse_mode': 'Markdown'})
    
    # Kirim Pesan 2 (Foto)
    for path in chart_paths:
        if os.path.exists(path):
            with open(path, 'rb') as photo:
                requests.post(url_photo, data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': photo})
                
    # Kirim Pesan 3 (Diagnostik)
    requests.post(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_diag, 'parse_mode': 'Markdown'})

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    start_time = time.time()
    sekarang_wita = datetime.now(pytz.timezone('Asia/Makassar'))
    
    try:
        # 1. Fetch Data
        df, kurs_idr, indodax_live_idr = fetch_data_with_retry()
        df, features = engineer_features(df)
        
        latest_close_usd = df['Close'].iloc[-1]
        volatility = df["Volatility"].iloc[-1]
        current_atr = df["ATR"].iloc[-1]
        news_status = fetch_crypto_news_sentiment()
        
        # 2. Latih Dual Engine
        try:
            prob_1h, acc_1h = train_honest_model(df, features, "Target_1H", shift_len=1)
        except Exception as e:
            diagnostics["ai_1h"] = f"❌ Error ML 1H ({str(e)[:20]})"
            prob_1h, acc_1h = 0.5, 0
            
        try:
            prob_6h, acc_6h = train_honest_model(df, features, "Target_6H", shift_len=6)
        except Exception as e:
            diagnostics["ai_6h"] = f"❌ Error ML 6H ({str(e)[:20]})"
            prob_6h, acc_6h = 0.5, 0

        # 3. Database & Evaluasi
        t_1h, p_1h, eval_1h, t_6h, p_6h, eval_6h, risk_mult = manage_history_and_evaluate(df, prob_1h, prob_6h)
        
        # 4. Manajemen Risiko
        exposure = position_sizing_kelly(prob_1h, prob_6h, current_atr, risk_mult)
        exp_1h_usd, var95_usd = monte_carlo_simulation(latest_close_usd, volatility, steps=1) # SL 1 Jam
        var95_idr = var95_usd * kurs_idr
        
        # --- SUSUN LAPORAN ---
        def get_arah(prob):
            if prob >= 0.6: return "NAIK KUAT 🚀"
            elif prob > 0.5: return "Cenderung NAIK 📈"
            elif prob <= 0.4: return "TURUN KUAT 🚨"
            else: return "Cenderung TURUN 📉"
            
        arah_1h = get_arah(prob_1h)
        arah_6h = get_arah(prob_6h)
        
        # Logika Sinyal Kesimpulan
        if prob_1h > 0.5 and prob_6h > 0.5: kesimpulan = "Tren utama NAIK, jangka pendek juga NAIK. Momentum sangat bagus (STRONG BUY)."
        elif prob_1h <= 0.5 and prob_6h > 0.5: kesimpulan = "Tren utama NAIK, tapi 1 jam ke depan KOREKSI TURUN. Waktu bagus untuk Buy the Dip (Cicil Bawah)."
        elif prob_1h > 0.5 and prob_6h <= 0.5: kesimpulan = "Ada pantulan NAIK sementara, tapi tren 6 jam masih TURUN. Rawan jebakan (Hindari/Sell on Strength)."
        else: kesimpulan = "Pasar sedang HANCUR. Jangka pendek dan panjang kompak TURUN. (STRONG SELL / WAIT)."

        # Teks Utama
        pesan_utama = f"💎 *LAPORAN TRADING AI GODMODE (DUAL ENGINE)* 💎\n"
        pesan_utama += f"_{sekarang_wita.strftime('%d %B %Y | %H:%M WITA')}_\n\n"
        pesan_utama += f"💰 *Harga BTC Sekarang:* {format_rupiah(indodax_live_idr)}\n\n"
        
        pesan_utama += f"🎯 *PREDIKSI TAKTIS (1 JAM KE DEPAN):*\n"
        pesan_utama += f"├ Arah: *{arah_1h}* (Keyakinan: {prob_1h*100:.1f}%)\n"
        pesan_utama += f"└ Cek 1 Jam Lalu: {eval_1h}\n\n"
        
        pesan_utama += f"🔭 *PREDIKSI TREN (6 JAM KE DEPAN):*\n"
        pesan_utama += f"├ Arah: *{arah_6h}* (Keyakinan: {prob_6h*100:.1f}%)\n"
        pesan_utama += f"└ Cek 6 Jam Lalu: {eval_6h}\n\n"
        
        pesan_utama += f"🚦 *KESIMPULAN SINYAL:*\n_{kesimpulan}_\n\n"
        
        pesan_utama += f"📊 *MANAJEMEN MODAL & REM OTOMATIS:*\n"
        if risk_mult < 1.0:
            pesan_utama += f"├ ⚠️ *STATUS:* REM DARURAT AKTIF (Akurasi AI Turun!)\n"
        pesan_utama += f"├ Alokasi Modal Aman: Maksimal *{exposure}%* saldo.\n"
        pesan_utama += f"└ Batas Apes (SL 95%): {format_rupiah(var95_idr)}\n\n"
        
        pesan_utama += f"📰 *SENTIMEN BERITA:* {news_status}"

        # Teks Diagnostik
        waktu_eksekusi = time.time() - start_time
        global_status = "🟢 *BOT BERJALAN NORMAL 100%*" if all("✅" in v for v in diagnostics.values()) else "🟡 *BERJALAN DENGAN PERINGATAN*"
        
        pesan_diag = f"🛠️ *DIAGNOSTIK & KESEHATAN SISTEM* 🛠️\n\n"
        pesan_diag += f"🌐 Koneksi Data: {diagnostics['api']}\n"
        pesan_diag += f"🗄️ Database CSV: {diagnostics['csv']}\n"
        pesan_diag += f"🧠 Mesin AI 1H: {diagnostics['ai_1h']}\n"
        pesan_diag += f"🧠 Mesin AI 6H: {diagnostics['ai_6h']}\n"
        pesan_diag += f"⏱️ Waktu Proses: {waktu_eksekusi:.1f} detik\n\n"
        pesan_diag += f"Status Global: {global_status}"

        # Buat Chart
        plot_professional_analysis(df, t_1h, p_1h, t_6h, p_6h, "chart_main.png")
        plot_zoomed_analysis(df, t_1h, p_1h, t_6h, p_6h, "chart_zoom.png")
        
        # Kirim
        send_telegram_messages(pesan_utama, ["chart_main.png", "chart_zoom.png"], pesan_diag)

    except Exception as fatal_e:
        # Jika benar-benar Kiamat API (Retry gagal semua)
        pesan_fatal = f"🚨 *BOT MATI MENDADAK (FATAL ERROR)* 🚨\n\n"
        pesan_fatal += f"Penyebab: {str(fatal_e)}\n"
        pesan_fatal += f"Waktu: {sekarang_wita.strftime('%H:%M WITA')}\n\n"
        pesan_fatal += f"💡 _Sistem otomatis menyerah untuk menghemat kuota GitHub. Silakan cek koneksi Indodax/Yahoo._"
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_fatal, 'parse_mode': 'Markdown'})

if __name__ == "__main__":
    main()