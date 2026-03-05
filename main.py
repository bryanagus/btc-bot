# ==============================================================================
# BTC QUANT GODMODE PRO MAX - TAHAP 1 (WEB3 CORE ENGINE FULL VERSION)
# Fitur: Fast Backoff Retry, Auto-Risk, System Diagnostics, Zero Data Leakage
# Indikator Full: ADX, MACD, Trend Slope, Momentum, Regime, VWAP, Bollinger
# Visual: 3 Grafik Lengkap (Main, Zoom, Dashboard Indikator)
# Web3 Ready: Evaluasi Murni Indodax IDR & Auto-Generate dashboard_data.json
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
import json
from datetime import datetime

# Import Library ML dan Finance
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 

# ================= KONFIGURASI =================
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
HISTORY_FILE = "ai_history_log.csv" 
WEB_DATA_FILE = "dashboard_data.json" # File Database untuk Web3
# ===============================================

# Global Diagnostics Dictionary
diagnostics = {
    "api": "✅ Normal (API Terhubung)",
    "csv": "✅ Sinkron (Riwayat Terbaca)",
    "ai_1h": "✅ Optimal",
    "ai_6h": "✅ Optimal",
    "chart": "✅ Optimal (3 Grafik Tercetak)",
    "web3": "✅ JSON Terupdate"
}

def format_rupiah(angka):
    if pd.isna(angka): return "Rp 0"
    return f"Rp {angka:,.0f}".replace(',', '.')

def format_usd(angka):
    if pd.isna(angka): return "$0"
    return f"${angka:,.0f}"

# ------------------------------------------------------------------------------
# 1. MODUL FETCH DATA & NEWS
# ------------------------------------------------------------------------------
def fetch_data_with_retry(period='90d', interval='1h'):
    print("[*] Mencoba menarik data dengan Fast Backoff...")
    delays = [5, 15, 30, 60]
    
    for attempt, delay in enumerate(delays + [0]):
        try:
            df = yf.download('BTC-USD', period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.droplevel(1)
            if df.index.tzinfo is None: 
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Makassar')
            
            idr_data = yf.download('IDR=X', period='5d', progress=False)
            if isinstance(idr_data.columns, pd.MultiIndex): 
                idr_data.columns = idr_data.columns.droplevel(1)
            kurs_idr = float(idr_data['Close'].dropna().iloc[-1])
            
            indodax_req = requests.get('https://indodax.com/api/ticker/btcidr', timeout=10).json()
            indodax_live_idr = float(indodax_req['ticker']['last'])
            
            return df, kurs_idr, indodax_live_idr
            
        except Exception as e:
            if attempt == len(delays):
                diagnostics["api"] = "❌ Gagal Terhubung (Server API Down)"
                raise Exception(f"API Timeout: {e}")
            time.sleep(delay)

def fetch_crypto_news_sentiment():
    try:
        rss_urls = ['https://www.coindesk.com/arc/outboundfeeds/rss/', 'https://cointelegraph.com/rss']
        analyzer = SentimentIntensityAnalyzer()
        crypto_lexicon = {"bullish": 2.5, "bearish": -2.5, "rekt": -3.0, "moon": 2.5, "pump": 2.0, "dump": -2.5}
        analyzer.lexicon.update(crypto_lexicon)
        
        compound_scores = []
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
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
# 2. FEATURE ENGINEERING (FULL INDICATORS)
# ------------------------------------------------------------------------------
def engineer_features(df):
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
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
    
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['ADX'] = (100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])).ewm(alpha=1/14, adjust=False).mean()

    if 'Volume' in df.columns:
        df['VWAP_24'] = (((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']).rolling(24).sum() / df['Volume'].rolling(24).sum()
    else:
        df['VWAP_24'] = df['Close']

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA_Spread"] = (df["EMA20"] - df["EMA50"]) / df["Close"]
    
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD_Line"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]
    
    df["Return_1H"] = df["Close"].pct_change()
    df["Volatility"] = df["Return_1H"].rolling(24).std()
    df["Trend_Slope"] = df["Close"].rolling(12).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)==12 else 0, raw=True)
    df["Momentum_Accel"] = df["Return_1H"].diff()
    df["Regime"] = np.where(df["EMA20"] > df["EMA50"], 1, 0)
    
    df["Target_1H"] = (df["Close"].shift(-1) > df["Close"]).astype(float)
    df["Target_6H"] = (df["Close"].shift(-6) > df["Close"]).astype(float)
    
    df = df.iloc[:-1] 
    features_cols = ["EMA_Spread", "RSI", "MACD_Hist", "Return_1H", "Volatility", "Trend_Slope", "Momentum_Accel", "Regime", "ADX"]
    return df, features_cols

# ------------------------------------------------------------------------------
# 3. MODUL AI JUJUR (DUAL TARGET)
# ------------------------------------------------------------------------------
def train_honest_model(df, features, target_col, shift_len):
    train_df = df.iloc[:-shift_len].dropna(subset=features + [target_col])
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    X_live = df[features].iloc[-1:].fillna(0).values

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
    
    return (prob_lr + prob_rf + prob_gb) / 3

# ------------------------------------------------------------------------------
# 4. DATABASE CSV & EVALUASI MURNI INDODAX (CORE FIX WEB3)
# ------------------------------------------------------------------------------
def manage_history_and_evaluate(df, prob_1h, prob_6h, indodax_live_idr):
    current_time = df.index[-1]
    target_time_1h = current_time + pd.Timedelta(hours=1)
    target_time_6h = current_time + pd.Timedelta(hours=6)
    
    eval_1h_msg = "Menunggu data..."
    eval_6h_msg = "Menunggu data..."
    risk_multiplier = 1.0 
    
    file_exists = os.path.isfile(HISTORY_FILE)
    
    # SYSTEM SAPU BERSIH: Hapus CSV lama jika formatnya salah (Bukan Web3 Format)
    if file_exists:
        try:
            with open(HISTORY_FILE, 'r') as f:
                header = f.readline()
            if 'indodax_start_price' not in header:
                print("[!] Format CSV lama terdeteksi. Mereset untuk Web3...")
                os.remove(HISTORY_FILE)
                file_exists = False
        except: pass
            
    try:
        if file_exists:
            history = pd.read_csv(HISTORY_FILE)
            history['timestamp_target_1h'] = pd.to_datetime(history['timestamp_target_1h'], utc=True).dt.tz_convert('Asia/Makassar')
            history['timestamp_target_6h'] = pd.to_datetime(history['timestamp_target_6h'], utc=True).dt.tz_convert('Asia/Makassar')
            
            # --- EVALUASI 1 JAM (MURNI INDODAX IDR) ---
            row_1h = history[history['timestamp_target_1h'] == current_time]
            if not row_1h.empty:
                past_prob = row_1h['prob_1h'].values[0]
                past_idr_price = float(row_1h['indodax_start_price'].values[0])
                arah_pred = "NAIK" if past_prob > 0.5 else "TURUN"
                arah_asli = "Naik" if indodax_live_idr > past_idr_price else "Turun"
                
                if (past_prob > 0.5 and indodax_live_idr > past_idr_price) or (past_prob <= 0.5 and indodax_live_idr <= past_idr_price):
                    eval_1h_msg = f"BENAR ✅ (Nebak {arah_pred}, Asli {arah_asli})"
                else:
                    eval_1h_msg = f"SALAH ❌ (Nebak {arah_pred}, Asli {arah_asli})"
                    
                history.loc[history['timestamp_target_1h'] == current_time, 'result_1h'] = eval_1h_msg
                history.loc[history['timestamp_target_1h'] == current_time, 'indodax_end_price_1h'] = indodax_live_idr

            # --- EVALUASI 6 JAM (MURNI INDODAX IDR) ---
            row_6h = history[history['timestamp_target_6h'] == current_time]
            if not row_6h.empty:
                past_prob = row_6h['prob_6h'].values[0]
                past_idr_price = float(row_6h['indodax_start_price'].values[0])
                arah_pred = "NAIK" if past_prob > 0.5 else "TURUN"
                arah_asli = "Naik" if indodax_live_idr > past_idr_price else "Turun"
                
                if (past_prob > 0.5 and indodax_live_idr > past_idr_price) or (past_prob <= 0.5 and indodax_live_idr <= past_idr_price):
                    eval_6h_msg = f"BENAR ✅ (Nebak {arah_pred})"
                else:
                    eval_6h_msg = f"SALAH ❌ (Nebak {arah_pred})"
                    
                history.loc[history['timestamp_target_6h'] == current_time, 'result_6h'] = eval_6h_msg
                history.loc[history['timestamp_target_6h'] == current_time, 'indodax_end_price_6h'] = indodax_live_idr

            history.to_csv(HISTORY_FILE, index=False)

            # AUTO RISK CALCULATION
            recent_1h = history.dropna(subset=['result_1h']).tail(5)
            if len(recent_1h) >= 3:
                benar_count = recent_1h['result_1h'].str.contains('BENAR').sum()
                win_rate = benar_count / len(recent_1h)
                if win_rate < 0.4: risk_multiplier = 0.1     
                elif win_rate < 0.6: risk_multiplier = 0.5   
                else: risk_multiplier = 1.0                  
                
    except Exception as e:
        diagnostics["csv"] = f"❌ Error Baca CSV ({str(e)[:20]})"
        risk_multiplier = 1.0

    # SIMPAN PREDIKSI BARU KE DALAM CSV
    try:
        with open(HISTORY_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['created_at', 'indodax_start_price', 'timestamp_target_1h', 'prob_1h', 'indodax_end_price_1h', 'result_1h', 'timestamp_target_6h', 'prob_6h', 'indodax_end_price_6h', 'result_6h'])
            writer.writerow([current_time.isoformat(), indodax_live_idr, target_time_1h.isoformat(), prob_1h, '', '', target_time_6h.isoformat(), prob_6h, '', ''])
    except:
        diagnostics["csv"] = "❌ Gagal Menulis CSV"

    return eval_1h_msg, eval_6h_msg, risk_multiplier

# ------------------------------------------------------------------------------
# 5. GENERATE DATA UNTUK WEB3 FRONTEND (JSON)
# ------------------------------------------------------------------------------
def generate_web3_dashboard_data(indodax_live_idr, prob_1h, prob_6h, current_atr, risk_mult, df):
    try:
        if not os.path.isfile(HISTORY_FILE): return
        
        history = pd.read_csv(HISTORY_FILE)
        
        # Ekstrak Riwayat 1 Jam (Maksimal 100 terakhir)
        hist_1h = history.dropna(subset=['result_1h']).tail(100)
        table_1h = []
        for _, row in hist_1h.iterrows():
            table_1h.append({
                "waktu_prediksi": pd.to_datetime(row['created_at']).strftime('%d %b %H:%M'),
                "waktu_target": pd.to_datetime(row['timestamp_target_1h']).strftime('%d %b %H:%M'),
                "harga_awal": row['indodax_start_price'],
                "harga_akhir": row['indodax_end_price_1h'],
                "prediksi": "NAIK" if row['prob_1h'] > 0.5 else "TURUN",
                "keyakinan": f"{row['prob_1h']*100:.1f}%",
                "status": row['result_1h'].split(' ')[0] # Ambil kata BENAR/SALAH saja
            })
            
        # Ekstrak Riwayat 6 Jam (Maksimal 100 terakhir)
        hist_6h = history.dropna(subset=['result_6h']).tail(100)
        table_6h = []
        for _, row in hist_6h.iterrows():
            table_6h.append({
                "waktu_prediksi": pd.to_datetime(row['created_at']).strftime('%d %b %H:%M'),
                "waktu_target": pd.to_datetime(row['timestamp_target_6h']).strftime('%d %b %H:%M'),
                "harga_awal": row['indodax_start_price'],
                "harga_akhir": row['indodax_end_price_6h'],
                "prediksi": "NAIK" if row['prob_6h'] > 0.5 else "TURUN",
                "keyakinan": f"{row['prob_6h']*100:.1f}%",
                "status": row['result_6h'].split(' ')[0]
            })

        total_benar_1h = len([x for x in table_1h if "BENAR" in x['status']])
        total_benar_6h = len([x for x in table_6h if "BENAR" in x['status']])
        win_rate_1h = round((total_benar_1h / len(table_1h) * 100) if table_1h else 0, 1)
        win_rate_6h = round((total_benar_6h / len(table_6h) * 100) if table_6h else 0, 1)

        web_data = {
            "last_update": datetime.now(pytz.timezone('Asia/Makassar')).strftime('%d %B %Y %H:%M WITA'),
            "live_price_idr": indodax_live_idr,
            "current_prediction": {
                "prob_1h": prob_1h,
                "prob_6h": prob_6h,
                "arah_1h": "NAIK" if prob_1h > 0.5 else "TURUN",
                "arah_6h": "NAIK" if prob_6h > 0.5 else "TURUN"
            },
            "stats": {
                "win_rate_1h": win_rate_1h,
                "win_rate_6h": win_rate_6h,
                "risk_multiplier": risk_mult
            },
            "indicators": {
                "rsi": round(df['RSI'].iloc[-1], 2),
                "adx": round(df['ADX'].iloc[-1], 2),
                "trend": "NAIK" if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else "TURUN"
            },
            "history_1h": table_1h[::-1], # Urutan dari yang paling baru
            "history_6h": table_6h[::-1]
        }

        with open(WEB_DATA_FILE, 'w') as f:
            json.dump(web_data, f, indent=4)
            
    except Exception as e:
        diagnostics["web3"] = f"❌ Gagal Export JSON ({str(e)[:20]})"

# ------------------------------------------------------------------------------
# 6. RISK MANAGEMENT & VISUALIZATION (FULL 3 CHARTS)
# ------------------------------------------------------------------------------
def monte_carlo_simulation(price, vol, steps=1, sims=2000):
    paths = []
    for _ in range(sims):
        prices = [price]
        for _ in range(steps): prices.append(prices[-1] * (1 + np.random.normal(0, vol)))
        paths.append(prices)
    final_prices = np.array(paths)[:, -1]
    return np.mean(final_prices), np.percentile(final_prices, 5)

def position_sizing_kelly(prob_1h, prob_6h, atr, risk_multiplier):
    avg_prob = (prob_1h + prob_6h) / 2
    if avg_prob < 0.5: return 0.0
    reward, risk = atr * 3.5, atr * 1.5
    if risk <= 0: return 0.0 
    edge = avg_prob - ((1 - avg_prob) / (reward/risk))
    size = min(max(edge, 0), 0.25) * risk_multiplier 
    return round(size * 100, 2)

def plot_professional_analysis(df, prob_1h, prob_6h, atr, filename="chart_main.png"):
    plot_data = df.tail(80) 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 8))
    
    plt.plot(plot_data.index, plot_data['Close'], label='Harga Asli BTC', color='black', linewidth=2, zorder=5)
    plt.plot(plot_data.index, plot_data['VWAP_24'], label='Garis Bandar (VWAP)', color='#ff7f0e', linestyle='-.')
    
    last_time = plot_data.index[-1]
    last_price = plot_data['Close'].iloc[-1]
    
    t_1h = last_time + pd.Timedelta(hours=1)
    p_1h = last_price + (atr * (prob_1h - 0.5) * 2)
    t_6h = last_time + pd.Timedelta(hours=6)
    p_6h = last_price + (atr * (prob_6h - 0.5) * 4)

    plt.scatter(t_1h, p_1h, color='blue', s=80, marker='o', zorder=10, label='Target 1 Jam')
    plt.plot([last_time, t_1h], [last_price, p_1h], color='blue', linestyle='--', linewidth=2)
    plt.scatter(t_6h, p_6h, color='red', s=100, marker='X', zorder=10, label='Target 6 Jam')
    plt.plot([last_time, t_6h], [last_price, p_6h], color='red', linestyle='--', linewidth=2)

    plt.title('GRAFIK UTAMA AI DUAL ENGINE (80 JAM)', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', framealpha=0.9)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_zoomed_analysis(df, prob_1h, prob_6h, atr, filename="chart_zoom.png"):
    plot_data = df.tail(12) 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(plot_data.index, plot_data['Close'], color='black', linewidth=3)
    ax1.plot(plot_data.index, plot_data['VWAP_24'], color='#ff7f0e', linestyle='-.', linewidth=2)
    ax1.fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], color='gray', alpha=0.15)

    last_time = plot_data.index[-1]
    last_price = plot_data['Close'].iloc[-1]
    
    t_1h = last_time + pd.Timedelta(hours=1)
    p_1h = last_price + (atr * (prob_1h - 0.5) * 2)
    t_6h = last_time + pd.Timedelta(hours=6)
    p_6h = last_price + (atr * (prob_6h - 0.5) * 4)
    
    ax1.scatter(t_1h, p_1h, color='blue', s=100)
    ax1.plot([last_time, t_1h], [last_price, p_1h], color='blue', linestyle='--')
    ax1.scatter(t_6h, p_6h, color='red', s=120, marker='X')
    ax1.plot([last_time, t_6h], [last_price, p_6h], color='red', linestyle='--')

    ax1.annotate(f"Harga: {format_usd(last_price)}", (last_time, last_price), xytext=(0, -25), textcoords='offset points', ha='center', bbox=dict(boxstyle="round", fc="black", alpha=0.8), color="white")
    ax1.set_title('🔍 ZOOM PITA BOLLINGER & TARGET GANDA', fontsize=16, fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_dashboard_indicators(df, filename="chart_indicators.png"):
    try:
        plot_data = df.tail(80)
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 16))
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(plot_data.index, plot_data['Close'], color='black', label='Harga BTC', linewidth=2)
        ax1.plot(plot_data.index, plot_data['EMA20'], color='blue', alpha=0.8, label='EMA 20')
        ax1.plot(plot_data.index, plot_data['EMA50'], color='red', alpha=0.8, label='EMA 50')
        ax1.set_title('1. TREN HARGA & EMA', fontweight='bold')
        ax1.legend(loc='upper left')

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(plot_data.index, plot_data['RSI'], color='purple', linewidth=2, label='RSI')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.set_title('2. RSI (Momentum)', fontweight='bold')
        ax2.legend(loc='upper left')

        ax3 = fig.add_subplot(gs[2, 0])
        ax3.bar(plot_data.index, plot_data['MACD_Hist'], color=np.where(plot_data['MACD_Hist']>0, 'green', 'red'), alpha=0.5)
        ax3.plot(plot_data.index, plot_data['MACD_Line'], color='blue', label='MACD')
        ax3.plot(plot_data.index, plot_data['MACD_Signal'], color='orange', label='Signal')
        ax3.set_title('3. MACD', fontweight='bold')
        ax3.legend(loc='upper left')

        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(plot_data.index, plot_data['ADX'], color='black', linewidth=2, label='ADX')
        ax4.plot(plot_data.index, plot_data['+DI'], color='green', alpha=0.8, label='+DI')
        ax4.plot(plot_data.index, plot_data['-DI'], color='red', alpha=0.8, label='-DI')
        ax4.set_title('4. ADX & DMI', fontweight='bold')
        ax4.legend(loc='upper left')

        for ax in [ax1, ax2, ax3, ax4]: ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
    except Exception as e:
        diagnostics["chart"] = f"❌ Gagal Render Dashboard ({str(e)[:20]})"

# ------------------------------------------------------------------------------
# 7. TELEGRAM SENDER
# ------------------------------------------------------------------------------
def send_telegram_messages(pesan_utama, chart_paths, pesan_diag):
    if not TELEGRAM_BOT_TOKEN: return
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    requests.post(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_utama, 'parse_mode': 'Markdown'})
    for path in chart_paths:
        if os.path.exists(path):
            with open(path, 'rb') as photo:
                requests.post(url_photo, data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': photo})
    requests.post(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_diag, 'parse_mode': 'Markdown'})

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    start_time = time.time()
    sekarang_wita = datetime.now(pytz.timezone('Asia/Makassar'))
    
    try:
        df, kurs_idr, indodax_live_idr = fetch_data_with_retry()
        df, features = engineer_features(df)
        
        latest_close_usd = df['Close'].iloc[-1]
        volatility = df["Volatility"].iloc[-1]
        current_atr = df["ATR"].iloc[-1]
        news_status = fetch_crypto_news_sentiment()
        
        tren_status = "Kuat (Pasar sedang aktif bergerak)" if df['ADX'].iloc[-1] > 25 else "Lemah / Sideways (Pasar ragu-ragu)"
        vwap_status = "Aman (Harga di atas rata-rata tarikan bandar)" if latest_close_usd > df['VWAP_24'].iloc[-1] else "Bahaya (Harga di bawah rata-rata tarikan bandar)"
        
        try: prob_1h = train_honest_model(df, features, "Target_1H", shift_len=1)
        except Exception as e: diagnostics["ai_1h"] = f"❌ Error 1H ({str(e)[:15]})"; prob_1h = 0.5
            
        try: prob_6h = train_honest_model(df, features, "Target_6H", shift_len=6)
        except Exception as e: diagnostics["ai_6h"] = f"❌ Error 6H ({str(e)[:15]})"; prob_6h = 0.5

        # EVALUASI MURNI RUPIAH INDODAX
        eval_1h, eval_6h, risk_mult = manage_history_and_evaluate(df, prob_1h, prob_6h, indodax_live_idr)
        
        # GENERATE JSON UNTUK WEB3 FRONTEND
        generate_web3_dashboard_data(indodax_live_idr, prob_1h, prob_6h, current_atr, risk_mult, df)
        
        exposure = position_sizing_kelly(prob_1h, prob_6h, current_atr, risk_mult)
        exp_1h_usd, var95_usd = monte_carlo_simulation(latest_close_usd, volatility, steps=1) 
        var95_idr = var95_usd * kurs_idr
        
        def get_arah(prob):
            if prob >= 0.6: return "NAIK KUAT 🚀"
            elif prob > 0.5: return "Cenderung NAIK 📈"
            elif prob <= 0.4: return "TURUN KUAT 🚨"
            else: return "Cenderung TURUN 📉"
            
        arah_1h = get_arah(prob_1h)
        arah_6h = get_arah(prob_6h)
        
        if prob_1h > 0.5 and prob_6h > 0.5: kesimpulan = "Tren utama NAIK, jangka pendek juga NAIK. Momentum sangat bagus (STRONG BUY)."
        elif prob_1h <= 0.5 and prob_6h > 0.5: kesimpulan = "Tren utama NAIK, tapi 1 jam ke depan KOREKSI TURUN. Waktu bagus untuk Buy the Dip (Cicil Bawah)."
        elif prob_1h > 0.5 and prob_6h <= 0.5: kesimpulan = "Ada pantulan NAIK sementara, tapi tren 6 jam masih TURUN. Rawan jebakan (Hindari/Sell on Strength)."
        else: kesimpulan = "Pasar sedang HANCUR. Jangka pendek dan panjang kompak TURUN. (STRONG SELL / WAIT)."

        # LAPORAN TELEGRAM FULL (SAMA SEPERTI VERSI SEBELUMNYA)
        pesan_utama = f"💎 *LAPORAN TRADING AI GODMODE (WEB3 ENGINE)* 💎\n"
        pesan_utama += f"_{sekarang_wita.strftime('%d %B %Y | %H:%M WITA')}_\n\n"
        pesan_utama += f"💰 *Harga BTC Sekarang:* {format_rupiah(indodax_live_idr)}\n\n"
        
        pesan_utama += f"🎯 *PREDIKSI TAKTIS (1 JAM KE DEPAN):*\n"
        pesan_utama += f"├ Arah: *{arah_1h}* (Keyakinan: {prob_1h*100:.1f}%)\n"
        pesan_utama += f"└ Cek 1 Jam Lalu: {eval_1h}\n\n"
        
        pesan_utama += f"🔭 *PREDIKSI TREN (6 JAM KE DEPAN):*\n"
        pesan_utama += f"├ Arah: *{arah_6h}* (Keyakinan: {prob_6h*100:.1f}%)\n"
        pesan_utama += f"└ Cek 6 Jam Lalu: {eval_6h}\n\n"
        
        pesan_utama += f"🚦 *KESIMPULAN SINYAL:*\n_{kesimpulan}_\n\n"
        
        pesan_utama += f"📊 *MANAJEMEN MODAL & PASAR:*\n"
        if risk_mult < 1.0:
            pesan_utama += f"├ ⚠️ *STATUS:* REM DARURAT AKTIF (Akurasi AI Turun!)\n"
        pesan_utama += f"├ Alokasi Modal Aman: Maksimal *{exposure}%* saldo.\n"
        pesan_utama += f"├ Tren Saat Ini (ADX): {tren_status}\n"
        pesan_utama += f"├ Posisi Bandar (VWAP): {vwap_status}\n"
        pesan_utama += f"└ Batas Apes (SL 95%): {format_rupiah(var95_idr)}\n\n"
        
        pesan_utama += f"📰 *SENTIMEN BERITA:* {news_status}"

        # RENDER 3 GRAFIK LENGKAP
        plot_professional_analysis(df, prob_1h, prob_6h, current_atr, "chart_main.png")
        plot_zoomed_analysis(df, prob_1h, prob_6h, current_atr, "chart_zoom.png")
        plot_dashboard_indicators(df, "chart_indicators.png")
        
        # PESAN DIAGNOSTIK
        global_status = "🟢 *BOT BERJALAN NORMAL 100%*" if all("✅" in v for v in diagnostics.values()) else "🟡 *BERJALAN DENGAN PERINGATAN*"
        pesan_diag = f"🛠️ *DIAGNOSTIK & KESEHATAN SISTEM* 🛠️\n\n"
        for k, v in diagnostics.items(): pesan_diag += f"├ {v}\n"
        pesan_diag += f"├ ⏱️ Waktu Proses: {time.time() - start_time:.1f} detik\n\n"
        pesan_diag += f"Status Global: {global_status}"

        # KIRIM SEMUANYA KE TELEGRAM
        send_telegram_messages(pesan_utama, ["chart_main.png", "chart_zoom.png", "chart_indicators.png"], pesan_diag)

    except Exception as fatal_e:
        pesan_fatal = f"🚨 *BOT MATI MENDADAK (FATAL ERROR)* 🚨\n\nPenyebab: {str(fatal_e)}\nWaktu: {sekarang_wita.strftime('%H:%M WITA')}"
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_fatal})

if __name__ == "__main__":
    main()