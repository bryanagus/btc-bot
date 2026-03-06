# ==============================================================================
# BTC QUANT GODMODE PRO MAX - THE HYBRID ARBITRAGE EDITION
# Fitur: Fast Backoff, Auto-Risk, Monte Carlo, Kelly Sizing, 3 Full Charts
# Laporan: Rapi, Mudah Dimengerti, Smart Diagnostic, Persentase Perubahan, Saran Eksekusi
# AI Upgrade: Log Returns, Bollinger Squeeze, Normalized ATR
# PRO Upgrade (Tier 1 & 3): Fear & Greed API + Hyperparameter Tuning AI
# Sensor Waktu: Presisi Cronjob 10 Menitan (Jendela menit 05-19)
# Evaluasi Murni: USD Global (Yahoo Finance) agar AI tidak bingung.
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
import time
import csv
import json
import pickle
from datetime import datetime

# Import Library ML dan Finance
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 

# ================= KONFIGURASI =================
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
HISTORY_FILE = "ai_history_log.csv" 
WEB_DATA_FILE = "dashboard_data.json"
ALERT_FILE = "last_alert_state.json"
# ===============================================

diagnostics = {
    "api": "✅ Normal (API Terhubung)",
    "csv": "✅ Sinkron (Hybrid Format Terbaca)",
    "ai_1h": "✅ Optimal",
    "ai_6h": "✅ Optimal",
    "chart": "✅ Optimal (3 Grafik PNG & WebP Tercetak)",
    "web3": "✅ JSON Hybrid Terupdate"
}

def format_rupiah(angka):
    if pd.isna(angka): return "Rp 0"
    return f"Rp {angka:,.0f}".replace(',', '.')

def format_usd(angka):
    if pd.isna(angka): return "$0"
    return f"${angka:,.2f}"

# ------------------------------------------------------------------------------
# 1. MODUL FETCH DATA & NEWS (HYBRID, UTC ENGINE, ALTERNATIVE DATA)
# ------------------------------------------------------------------------------
def fetch_fear_and_greed():
    """ Mengambil Data Psikologi Pasar Dunia (Fear & Greed Index) """
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=100", timeout=10).json()
        fng_dict = {pd.to_datetime(x['timestamp'], unit='s', utc=True).date(): float(x['value']) for x in resp['data']}
        return fng_dict
    except Exception as e:
        diagnostics["api"] = diagnostics["api"].replace("✅ Normal", "🟡 Peringatan") + " | 🟡 F&G API Gagal"
        return {}

def fetch_data_with_retry(period='90d', interval='1h'):
    print("[*] Mencoba menarik data Hybrid dengan Fast Backoff...")
    delays = [5, 15, 30, 60]
    
    for attempt, delay in enumerate(delays + [0]):
        try:
            df = yf.download('BTC-USD', period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.droplevel(1)
            
            df.index = pd.to_datetime(df.index, utc=True)
            
            idr_data = yf.download('IDR=X', period='5d', progress=False)
            if isinstance(idr_data.columns, pd.MultiIndex): 
                idr_data.columns = idr_data.columns.droplevel(1)
            kurs_idr = float(idr_data['Close'].dropna().iloc[-1])
            
            try:
                indodax_req = requests.get('https://indodax.com/api/ticker/btcidr', timeout=10).json()
                indodax_live_idr = float(indodax_req['ticker']['last'])
            except Exception as e:
                indodax_live_idr = float(df['Close'].iloc[-1]) * kurs_idr
                diagnostics["api"] = "🟡 Ticker Indodax Down (Menggunakan Estimasi Kurs)"
            
            return df, kurs_idr, indodax_live_idr
            
        except Exception as e:
            if attempt == len(delays):
                diagnostics["api"] = "❌ Gagal Terhubung (Server API Global Down)"
                raise Exception(f"API Timeout: {e}")
            time.sleep(delay)

def fetch_indodax_depth():
    try:
        resp = requests.get('https://indodax.com/api/depth/btcidr', timeout=10).json()
        limit_buy = min(15, len(resp.get('buy', [])))
        limit_sell = min(15, len(resp.get('sell', [])))
        
        buy_wall = sum([float(x[0]) * float(x[1]) for x in resp.get('buy', [])[:limit_buy]])
        sell_wall = sum([float(x[0]) * float(x[1]) for x in resp.get('sell', [])[:limit_sell]])
        return buy_wall, sell_wall
    except Exception as e:
        diagnostics["api"] = diagnostics["api"].replace("✅", "🟡") + " | 🟡 Depth API Gagal"
        return 0, 0

def fetch_crypto_news_sentiment():
    try:
        rss_urls = ['https://www.coindesk.com/arc/outboundfeeds/rss/', 'https://cointelegraph.com/rss']
        analyzer = SentimentIntensityAnalyzer()
        crypto_lexicon = {"bullish": 2.5, "bearish": -2.5, "rekt": -3.0, "moon": 2.5, "pump": 2.0, "dump": -2.5}
        analyzer.lexicon.update(crypto_lexicon)
        
        compound_scores = []
        headers = {'User-Agent': 'Mozilla/5.0'}
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
# 2. FEATURE ENGINEERING (ADVANCED QUANT INDICATORS + F&G INJECTION)
# ------------------------------------------------------------------------------
def engineer_features(df, fng_dict):
    df = df.copy() 
    
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
    
    # Advanced Quant Features
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["ATR_Ratio"] = df["ATR"] / df["Close"]
    
    # Injection Fear & Greed ke Data AI
    df['Date_Only'] = df.index.date
    df['FnG_Index'] = df['Date_Only'].map(fng_dict)
    df['FnG_Index'] = df['FnG_Index'].ffill().bfill().fillna(50.0) 
    df.drop(columns=['Date_Only'], inplace=True)

    df["Target_1H"] = (df["Close"].shift(-1) > df["Close"]).astype(float)
    df["Target_6H"] = (df["Close"].shift(-6) > df["Close"]).astype(float)
    
    features_cols = [
        "EMA_Spread", "RSI", "MACD_Hist", "Log_Return", "Volatility", 
        "Trend_Slope", "Momentum_Accel", "Regime", "ADX", "BB_Width", "ATR_Ratio", "FnG_Index"
    ]
    return df, features_cols

# ------------------------------------------------------------------------------
# 3. MODUL AI JUJUR DENGAN CACHING & TIER 3 HYPERPARAMETER TUNING
# ------------------------------------------------------------------------------
def train_honest_model(df, features, target_col, shift_len, model_name):
    model_file = f"ai_model_{model_name}.pkl"
    need_training = True
    
    if os.path.exists(model_file):
        file_age = time.time() - os.path.getmtime(model_file)
        if file_age < 86400:
            need_training = False
            
    X_live = df[features].iloc[-1:].fillna(0).values

    if not need_training:
        try:
            print(f"[*] Memanggil Otak AI yang sudah pintar ({model_name})... Cepat!")
            with open(model_file, 'rb') as f:
                models = pickle.load(f)
            lr_cal = models['lr']
            rf = models['rf']
            gb = models['gb']
            scaler = models['scaler']
            X_live_scaled = scaler.transform(X_live)
        except Exception as e:
            print(f"[*] Gagal memanggil model lama, melatih ulang... ({e})")
            need_training = True

    if need_training:
        print(f"[*] Melatih ulang Otak AI ({model_name}) dengan Tier 3 Tuning...")
        train_df = df.iloc[: -(shift_len + 1)].dropna(subset=features + [target_col])
        X_train = train_df[features].values
        y_train = train_df[target_col].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_live_scaled = scaler.transform(X_live)

        # AI Tuning Parameters
        lr_base = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42)
        lr_cal = CalibratedClassifierCV(lr_base, method="sigmoid", cv=3)
        lr_cal.fit(X_train_scaled, y_train)
        
        rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
        gb.fit(X_train_scaled, y_train)
        
        with open(model_file, 'wb') as f:
            pickle.dump({'lr': lr_cal, 'rf': rf, 'gb': gb, 'scaler': scaler}, f)

    prob_lr = lr_cal.predict_proba(X_live_scaled)[0,1]
    prob_rf = rf.predict_proba(X_live_scaled)[0,1]
    prob_gb = gb.predict_proba(X_live_scaled)[0,1]
    
    return (prob_lr + prob_rf + prob_gb) / 3

# ------------------------------------------------------------------------------
# 4. DATABASE HYBRID, AUTO-HEALING & LAZY FETCHING INDODAX
# ------------------------------------------------------------------------------
def get_historical_indodax_price(t_target_utc, past_usd, past_idr, actual_usd):
    try:
        t_unix = int(t_target_utc.timestamp())
        url = f"https://indodax.com/tradingview/history?symbol=BTCIDR&resolution=60&from={t_unix-3600}&to={t_unix+3600}"
        resp = requests.get(url, timeout=10).json()
        if resp.get('s') == 'ok':
            times = resp.get('t', [])
            closes = resp.get('c', [])
            for i, ts in enumerate(times):
                if abs(ts - t_unix) <= 120:
                    return float(closes[i])
    except:
        pass
    
    past_ratio = past_idr / past_usd
    return actual_usd * past_ratio

def manage_history_and_evaluate(df, prob_1h, prob_6h, indodax_live_idr, kurs_idr):
    current_time = df.index[-1]
    usd_live_price = float(df['Close'].iloc[-1])
    target_time_1h = current_time + pd.Timedelta(hours=1)
    target_time_6h = current_time + pd.Timedelta(hours=6)
    
    eval_1h_msg = "Menunggu..."
    eval_6h_msg = "Menunggu..."
    risk_multiplier = 1.0 
    
    file_exists = os.path.isfile(HISTORY_FILE)
    
    if file_exists:
        try:
            with open(HISTORY_FILE, 'r') as f:
                header = f.readline()
            if 'FnG_Index' not in header and 'usd_start_price' not in header:
                os.remove(HISTORY_FILE)
                file_exists = False
        except: pass
            
    try:
        if file_exists:
            history = pd.read_csv(HISTORY_FILE)
            history['target_1h'] = pd.to_datetime(history['target_1h'], format='mixed', utc=True)
            history['target_6h'] = pd.to_datetime(history['target_6h'], format='mixed', utc=True)
            
            mask_1h = (history['target_1h'] <= current_time) & (history['result_1h'].isna())
            for idx, row in history[mask_1h].iterrows():
                t_target = row['target_1h']
                if t_target in df.index:
                    past_prob = float(row['prob_1h'])
                    past_usd_price = float(row['usd_start_price'])
                    past_idr_price = float(row['idr_start_price'])
                    actual_end_price = float(df.loc[t_target, 'Close'])
                    
                    arah_asli = "Naik" if actual_end_price > past_usd_price else "Turun"
                    if (past_prob > 0.5 and actual_end_price > past_usd_price) or (past_prob <= 0.5 and actual_end_price <= past_usd_price):
                        hasil = f"BENAR ✅ (Realita: {arah_asli})"
                    else:
                        hasil = f"SALAH ❌ (Realita: {arah_asli})"
                        
                    history.loc[idx, 'result_1h'] = hasil
                    history.loc[idx, 'usd_end_price_1h'] = actual_end_price
                    history.loc[idx, 'idr_end_price_1h'] = indodax_live_idr if t_target == current_time else get_historical_indodax_price(t_target, past_usd_price, past_idr_price, actual_end_price)
                    if t_target == current_time: eval_1h_msg = hasil

            mask_6h = (history['target_6h'] <= current_time) & (history['result_6h'].isna())
            for idx, row in history[mask_6h].iterrows():
                t_target = row['target_6h']
                if t_target in df.index:
                    past_prob = float(row['prob_6h'])
                    past_usd_price = float(row['usd_start_price'])
                    past_idr_price = float(row['idr_start_price'])
                    actual_end_price = float(df.loc[t_target, 'Close'])
                    
                    arah_asli = "Naik" if actual_end_price > past_usd_price else "Turun"
                    if (past_prob > 0.5 and actual_end_price > past_usd_price) or (past_prob <= 0.5 and actual_end_price <= past_usd_price):
                        hasil = f"BENAR ✅ (Realita: {arah_asli})"
                    else:
                        hasil = f"SALAH ❌ (Realita: {arah_asli})"
                        
                    history.loc[idx, 'result_6h'] = hasil
                    history.loc[idx, 'usd_end_price_6h'] = actual_end_price
                    history.loc[idx, 'idr_end_price_6h'] = indodax_live_idr if t_target == current_time else get_historical_indodax_price(t_target, past_usd_price, past_idr_price, actual_end_price)
                    if t_target == current_time: eval_6h_msg = hasil

            history.to_csv(HISTORY_FILE, index=False)

            recent_1h = history.dropna(subset=['result_1h']).tail(5)
            if len(recent_1h) >= 3:
                benar_count = recent_1h['result_1h'].str.contains('BENAR').sum()
                win_rate = benar_count / len(recent_1h)
                if win_rate < 0.4: risk_multiplier = 0.1     
                elif win_rate < 0.6: risk_multiplier = 0.5   
                else: risk_multiplier = 1.0                  
                
    except Exception as e:
        diagnostics["csv"] = f"❌ Error Baca/Update CSV: {e}"
        risk_multiplier = 1.0

    try:
        t_curr_str = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        t_1h_str = target_time_1h.strftime('%Y-%m-%dT%H:%M:%SZ')
        t_6h_str = target_time_6h.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        is_duplicate = False
        if file_exists:
            try:
                if len(history) > 0 and str(history['created_at'].iloc[-1]) == t_curr_str:
                    is_duplicate = True
            except: pass

        if not is_duplicate:
            with open(HISTORY_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow([
                        'created_at', 'kurs_usd_idr', 'usd_start_price', 'idr_start_price', 
                        'target_1h', 'prob_1h', 'usd_end_price_1h', 'idr_end_price_1h', 'result_1h',
                        'target_6h', 'prob_6h', 'usd_end_price_6h', 'idr_end_price_6h', 'result_6h'
                    ])
                writer.writerow([
                    t_curr_str, kurs_idr, usd_live_price, indodax_live_idr, 
                    t_1h_str, prob_1h, '', '', '', 
                    t_6h_str, prob_6h, '', '', ''
                ])
    except:
        diagnostics["csv"] = "❌ Gagal Menulis Baris Baru CSV"

    return eval_1h_msg, eval_6h_msg, risk_multiplier

# ------------------------------------------------------------------------------
# 5. GENERATE DATA WEB3 (TAMPILAN DIUBAH KE LOKAL/WITA)
# ------------------------------------------------------------------------------
def generate_web3_dashboard_data(indodax_idr, global_usd, kurs_idr, prob_1h, prob_6h, df, risk_mult):
    global_idr_converted = global_usd * kurs_idr
    spread_premium = indodax_idr - global_idr_converted
    
    try:
        if not os.path.isfile(HISTORY_FILE): return
        history = pd.read_csv(HISTORY_FILE)
        
        hist_1h = history.dropna(subset=['result_1h']).tail(100)
        table_1h = []
        for _, row in hist_1h.iterrows():
            waktu_local = pd.to_datetime(row['created_at'], format='mixed', utc=True).tz_convert('Asia/Makassar')
            table_1h.append({
                "waktu": waktu_local.strftime('%d %b %H:%M'),
                "start_usd": row['usd_start_price'],
                "end_usd": row['usd_end_price_1h'],
                "start_idr": row['idr_start_price'],
                "end_idr": row['idr_end_price_1h'],
                "prediksi": "NAIK" if row['prob_1h'] > 0.5 else "TURUN",
                "keyakinan": f"{row['prob_1h']*100:.1f}%",
                "status": row['result_1h'].split(' ')[0]
            })
            
        hist_6h = history.dropna(subset=['result_6h']).tail(100)
        table_6h = []
        for _, row in hist_6h.iterrows():
            waktu_local = pd.to_datetime(row['created_at'], format='mixed', utc=True).tz_convert('Asia/Makassar')
            table_6h.append({
                "waktu": waktu_local.strftime('%d %b %H:%M'),
                "start_usd": row['usd_start_price'],
                "end_usd": row['usd_end_price_6h'],
                "start_idr": row['idr_start_price'],
                "end_idr": row['idr_end_price_6h'],
                "prediksi": "NAIK" if row['prob_6h'] > 0.5 else "TURUN",
                "keyakinan": f"{row['prob_6h']*100:.1f}%",
                "status": row['result_6h'].split(' ')[0]
            })

        total_1h = len([x for x in table_1h if "BENAR" in x['status']])
        total_6h = len([x for x in table_6h if "BENAR" in x['status']])

        web_data = {
            "last_update": datetime.now(pytz.timezone('Asia/Makassar')).strftime('%d %B %Y %H:%M WITA'),
            "prices": {
                "indodax_idr": indodax_idr,
                "global_usd": global_usd,
                "kurs_idr": kurs_idr,
                "global_idr_converted": global_idr_converted,
                "spread_premium": spread_premium
            },
            "current_prediction": {
                "prob_1h": prob_1h,
                "prob_6h": prob_6h,
                "arah_1h": "NAIK" if prob_1h > 0.5 else "TURUN",
                "arah_6h": "NAIK" if prob_6h > 0.5 else "TURUN"
            },
            "stats": {
                "win_rate_1h": round((total_1h / len(table_1h) * 100) if table_1h else 0, 1),
                "win_rate_6h": round((total_6h / len(table_6h) * 100) if table_6h else 0, 1),
                "risk_multiplier": risk_mult
            },
            "indicators": {
                "rsi": round(df['RSI'].iloc[-1], 2),
                "adx": round(df['ADX'].iloc[-1], 2),
                "trend": "NAIK" if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else "TURUN"
            },
            "history_1h": table_1h[::-1],
            "history_6h": table_6h[::-1]
        }

        with open(WEB_DATA_FILE, 'w') as f:
            json.dump(web_data, f, indent=4)
            
    except Exception as e:
        diagnostics["web3"] = f"❌ Gagal Export JSON: {e}"

# ------------------------------------------------------------------------------
# 6. RISK MANAGEMENT & VISUALISASI
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

def plot_professional_analysis(df, prob_1h, prob_6h, atr, base_filename="chart_main"):
    plt.clf() 
    plot_data = df.tail(80).copy() 
    plot_data.index = plot_data.index.tz_convert('Asia/Makassar')
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 8))
    
    plt.plot(plot_data.index, plot_data['Close'], label='Harga Global (USD)', color='black', linewidth=2, zorder=5)
    plt.plot(plot_data.index, plot_data['VWAP_24'], label='VWAP (Bandar)', color='#ff7f0e', linestyle='-.')
    
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

    plt.title('GLOBAL MARKET DUAL ENGINE (80 HOURS) - EVALUATED IN USD', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', framealpha=0.9)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    plt.tight_layout()
    
    plt.savefig(f"{base_filename}.png", dpi=150)
    try: plt.savefig(f"{base_filename}.webp", format='webp', dpi=150)
    except: pass 
    plt.close('all')

def plot_zoomed_analysis(df, prob_1h, prob_6h, atr, base_filename="chart_zoom"):
    plt.clf()
    plot_data = df.tail(12).copy()
    plot_data.index = plot_data.index.tz_convert('Asia/Makassar')
    
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

    ax1.annotate(f"USD {last_price:,.0f}", (last_time, last_price), xytext=(0, -25), textcoords='offset points', ha='center', bbox=dict(boxstyle="round", fc="black", alpha=0.8), color="white")
    ax1.set_title('🔍 ZOOM BOLLINGER (USD GLOBAL)', fontsize=16, fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()
    
    plt.savefig(f"{base_filename}.png", dpi=150)
    try: plt.savefig(f"{base_filename}.webp", format='webp', dpi=150)
    except: pass
    plt.close('all')

def plot_dashboard_indicators(df, base_filename="chart_indicators"):
    try:
        plt.clf()
        plot_data = df.tail(80).copy()
        plot_data.index = plot_data.index.tz_convert('Asia/Makassar')
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 16))
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(plot_data.index, plot_data['Close'], color='black', label='Harga USD', linewidth=2)
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
        
        plt.savefig(f"{base_filename}.png", dpi=150)
        try: plt.savefig(f"{base_filename}.webp", format='webp', dpi=150)
        except: pass
        plt.close('all')
    except Exception as e:
        diagnostics["chart"] = f"❌ Gagal Render Dashboard: {e}"

# ------------------------------------------------------------------------------
# 7. TELEGRAM SENDER (DENGAN SMART DIAGNOSTIC ALERT)
# ------------------------------------------------------------------------------
def send_telegram_messages(pesan_utama, chart_paths, pesan_diag=""):
    if not TELEGRAM_BOT_TOKEN: return
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    requests.post(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_utama, 'parse_mode': 'Markdown'})
    for path in chart_paths:
        if os.path.exists(path):
            with open(path, 'rb') as photo:
                requests.post(url_photo, data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': photo})
                
    if pesan_diag.strip() != "":
        requests.post(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_diag, 'parse_mode': 'Markdown'})

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    start_time = time.time()
    sekarang_wita = datetime.now(pytz.timezone('Asia/Makassar'))
    
    try:
        # Panggil Fetch Data & Fear Greed Index
        fng_dict = fetch_fear_and_greed()
        df, kurs_idr, indodax_live_idr = fetch_data_with_retry()
        df, features = engineer_features(df, fng_dict)
        
        buy_wall, sell_wall = fetch_indodax_depth()
        
        global_usd = float(df['Close'].iloc[-1])
        global_idr_converted = global_usd * kurs_idr
        spread_premium = indodax_live_idr - global_idr_converted
        
        volatility = df["Volatility"].iloc[-1]
        current_atr = df["ATR"].iloc[-1]
        news_status = fetch_crypto_news_sentiment()
        
        tren_status = "Kuat" if df['ADX'].iloc[-1] > 25 else "Lemah / Sideways"
        vwap_status = "Aman (Harga di atas VWAP)" if global_usd > df['VWAP_24'].iloc[-1] else "Bahaya (Harga di bawah VWAP)"
        
        try: prob_1h = train_honest_model(df, features, "Target_1H", shift_len=1, model_name="1H")
        except Exception as e: diagnostics["ai_1h"] = f"❌ Error 1H: {e}"; prob_1h = 0.5
            
        try: prob_6h = train_honest_model(df, features, "Target_6H", shift_len=6, model_name="6H")
        except Exception as e: diagnostics["ai_6h"] = f"❌ Error 6H: {e}"; prob_6h = 0.5

        eval_1h, eval_6h, risk_mult = manage_history_and_evaluate(df, prob_1h, prob_6h, indodax_live_idr, kurs_idr)
        
        generate_web3_dashboard_data(indodax_live_idr, global_usd, kurs_idr, prob_1h, prob_6h, df, risk_mult)
        
        exposure = position_sizing_kelly(prob_1h, prob_6h, current_atr, risk_mult)
        exp_1h_usd, var95_usd = monte_carlo_simulation(global_usd, volatility, steps=1) 
        var95_idr = var95_usd * kurs_idr
        
        def get_arah(prob):
            if prob >= 0.6: return "NAIK KUAT 🚀"
            elif prob > 0.5: return "Cenderung NAIK 📈"
            elif prob <= 0.4: return "TURUN KUAT 🚨"
            else: return "Cenderung TURUN 📉"
            
        arah_1h = get_arah(prob_1h)
        arah_6h = get_arah(prob_6h)

        # LOGIKA SARAN EKSEKUSI (BELI/JUAL/TAHAN)
        fng_val = df['FnG_Index'].iloc[-1]
        saran_tindakan = ""
        alasan_saran = ""
        
        # Kondisi 1: Bullish Kuat + Indodax Diskon (Momen Emas Masuk)
        if prob_1h > 0.5 and prob_6h > 0.5 and spread_premium <= 0 and fng_val < 75:
            saran_tindakan = "🟢 STRONG BUY (BELI SEKARANG)"
            alasan_saran = "AI memprediksi tren global NAIK kuat, dan harga di Indodax saat ini SEDANG DISKON (lebih murah dari global). Ini adalah titik masuk (*entry*) yang sangat ideal."
        # Kondisi 2: Bullish tapi Indodax Sangat Mahal (Premium Tinggi)
        elif prob_1h > 0.5 and prob_6h > 0.5 and spread_premium > (global_idr_converted * 0.005):
            saran_tindakan = "🟡 TAHAN / JUAL SEBAGIAN (TAKE PROFIT)"
            alasan_saran = "Meski tren global diprediksi NAIK, harga di Indodax saat ini SANGAT MAHAL (Premium tinggi). Daripada beli di pucuk lokal, lebih bijak merealisasikan keuntungan (*take profit*) sebagian."
        # Kondisi 3: Bearish Kuat + Indodax Mahal (Waktunya Kabur)
        elif prob_1h <= 0.5 and prob_6h <= 0.5 and spread_premium > 0:
            saran_tindakan = "🔴 STRONG SELL (JUAL SEGERA)"
            alasan_saran = "AI memprediksi tren global TURUN tajam, namun harga Indodax saat ini masih ditawar mahal (Premium). Manfaatkan jeda harga ini untuk JUAL sebelum harga lokal ikut runtuh."
        # Kondisi 4: Bearish tapi Indodax Diskon (Tunggu Jemput Bawah)
        elif prob_1h <= 0.5 and prob_6h <= 0.5 and spread_premium < 0:
            saran_tindakan = "⚪ WAIT AND SEE (JANGAN BELI DULU)"
            alasan_saran = "Memang harga Indodax sedang diskon, tapi AI memprediksi harga global MASIH AKAN TURUN. Tahan peluru (*cash*) Anda, kita tunggu harga di titik dasar (*bottom*)."
        # Kondisi 5: Sideways / Ragu-ragu
        else:
            saran_tindakan = "⚪ NETRAL / TAHAN POSISI"
            alasan_saran = "Sinyal jangka pendek dan tren menengah sedang bertabrakan (pasar ragu-ragu). Tidak disarankan membuka posisi besar saat ini. Pantau pergerakan harga selanjutnya."

        # Peringatan F&G Ekstrem
        if fng_val >= 75:
            alasan_saran += "\n_⚠️ Peringatan: Pasar sedang Sangat Serakah (Extreme Greed). Waspada potensi bandar membanting harga tiba-tiba._"

        if buy_wall > sell_wall:
            status_orderbook = f"Tembok Beli Kuat 🟢 ({(buy_wall/sell_wall if sell_wall>0 else 1):.1f}x lipat dari Jual)"
        else:
            status_orderbook = f"Tembok Jual Kuat 🔴 ({(sell_wall/buy_wall if buy_wall>0 else 1):.1f}x lipat dari Beli)"

        ada_error_sistem = any("✅" not in v for v in diagnostics.values())

        # ====================================================================
        # SENSOR SILENT MODE, CRON KALKULASI PERSENTASE NAIK/TURUN
        # ====================================================================
        kirim_telegram = False
        alasan_kirim = ""
        is_routine_update = False
        teks_perubahan = "⚖️ Data Awal" 
        
        waktu_sekarang_str = sekarang_wita.strftime('%Y-%m-%d-%H')
        menit_sekarang = sekarang_wita.minute
        
        if not os.path.exists(ALERT_FILE):
            kirim_telegram = True
            alasan_kirim = "Bot Baru Dinyalakan / Sistem Reset"
            is_routine_update = True
        elif ada_error_sistem:
            kirim_telegram = True
            alasan_kirim = "⚠️ Terdeteksi Gangguan Sistem/API"
        else:
            try:
                with open(ALERT_FILE, 'r') as f:
                    last_state = json.load(f)
                
                # --- KALKULASI PERSENTASE HARGA ---
                last_price_saved = last_state.get('price', global_usd)
                if last_price_saved > 0:
                    raw_diff_pct = ((global_usd - last_price_saved) / last_price_saved) * 100
                else:
                    raw_diff_pct = 0.0
                    
                if raw_diff_pct > 0:
                    teks_perubahan = f"📈 Naik {raw_diff_pct:.2f}%"
                elif raw_diff_pct < 0:
                    teks_perubahan = f"📉 Turun {abs(raw_diff_pct):.2f}%"
                else:
                    teks_perubahan = f"⚖️ Stabil 0.00%"
                # -----------------------------------
                
                price_diff = abs(raw_diff_pct)
                selisih_waktu = time.time() - last_state.get('time', 0)
                last_hourly_report = last_state.get('last_hourly_report', '')
                
                # Syarat 1: Harga Gerak Ekstrem (> 1.5%)
                if price_diff >= 1.5:
                    kirim_telegram = True
                    alasan_kirim = f"Pergerakan Harga Drastis"
                
                # Syarat 2: Saran Eksekusi Berubah (Lebih berguna daripada hanya sinyal arah AI)
                elif saran_tindakan != last_state.get('saran', ''):
                    if selisih_waktu < 3600: 
                        kirim_telegram = False 
                    else:
                        kirim_telegram = True
                        alasan_kirim = f"Perubahan Rekomendasi Jual/Beli"
                
                # Syarat 3: Laporan Rutin Cronjob 10-Menitan (Trigger jika menit antara 05 s/d 19)
                elif 5 <= menit_sekarang < 20 and last_hourly_report != waktu_sekarang_str:
                    kirim_telegram = True
                    alasan_kirim = f"Laporan Rutin Update Market ({sekarang_wita.strftime('%H:10')} WITA)"
                    is_routine_update = True
                
                else:
                    kirim_telegram = False
            except Exception as e:
                kirim_telegram = True
                alasan_kirim = f"Reset Ingatan Sistem"
                is_routine_update = True

        plot_professional_analysis(df, prob_1h, prob_6h, current_atr, "chart_main")
        plot_zoomed_analysis(df, prob_1h, prob_6h, current_atr, "chart_zoom")
        plot_dashboard_indicators(df, "chart_indicators")
        
        if kirim_telegram:
            new_last_hourly = waktu_sekarang_str if is_routine_update else last_state.get('last_hourly_report', '')
            with open(ALERT_FILE, 'w') as f:
                json.dump({
                    'price': global_usd,
                    'signal': kesimpulan, # Tetap disimpan untuk kompatibilitas
                    'saran': saran_tindakan, # Disimpan untuk trigger jika berubah
                    'time': time.time(),
                    'last_hourly_report': new_last_hourly
                }, f)

            if fng_val >= 75: fng_str = f"{int(fng_val)} (Sangat Serakah 🤑)"
            elif fng_val >= 55: fng_str = f"{int(fng_val)} (Serakah 😋)"
            elif fng_val <= 25: fng_str = f"{int(fng_val)} (Sangat Takut 😱)"
            elif fng_val <= 45: fng_str = f"{int(fng_val)} (Takut 😨)"
            else: fng_str = f"{int(fng_val)} (Netral 😐)"

            # ============================================================
            # FORMAT LAPORAN TELEGRAM LENGKAP & RAPI 
            # ============================================================
            pesan_utama = f"💎 *LAPORAN TRADING AI (HYBRID USD-IDR)* 💎\n"
            pesan_utama += f"📅 {sekarang_wita.strftime('%d %B %Y | %H:%M WITA')}\n"
            pesan_utama += f"🔔 *Pemicu:* {alasan_kirim}\n"
            pesan_utama += f"⚡ *Pergerakan:* {teks_perubahan} (Sejak laporan terakhir)\n\n"
            
            pesan_utama += f"📊 *MONITOR HARGA & ARBITRASE*\n"
            pesan_utama += f"• Global (USD): {format_usd(global_usd)}\n"
            pesan_utama += f"• Kurs Dollar: {format_rupiah(kurs_idr)}\n"
            pesan_utama += f"• Harga Wajar (IDR): *{format_rupiah(global_idr_converted)}*\n"
            pesan_utama += f"• Indodax Live: *{format_rupiah(indodax_live_idr)}*\n"
            
            if spread_premium > 0:
                pesan_utama += f"• Selisih Indodax: {format_rupiah(abs(spread_premium))} (Lebih Mahal)\n"
            else:
                pesan_utama += f"• Selisih Indodax: {format_rupiah(abs(spread_premium))} (Lebih Murah)\n"
                
            pesan_utama += f"• Kondisi Pasar: {status_orderbook}\n\n"
            
            pesan_utama += f"🤖 *PREDIKSI AI (MACHINE LEARNING)*\n"
            pesan_utama += f"*1 Jam Kedepan (Taktis)*\n"
            pesan_utama += f"• Arah AI: *{arah_1h}* (Keyakinan: {prob_1h*100:.1f}%)\n"
            pesan_utama += f"• Akurasi 1 Jam Lalu: {eval_1h}\n\n"
            
            pesan_utama += f"*6 Jam Kedepan (Tren)*\n"
            pesan_utama += f"• Arah AI: *{arah_6h}* (Keyakinan: {prob_6h*100:.1f}%)\n"
            pesan_utama += f"• Akurasi 6 Jam Lalu: {eval_6h}\n\n"
            
            pesan_utama += f"🛡️ *INDIKATOR & SENTIMEN TERKINI*\n"
            if risk_mult < 1.0:
                pesan_utama += f"⚠️ *STATUS:* REM DARURAT AKTIF (Akurasi Bot Menurun)\n"
            pesan_utama += f"• Alokasi Dana Aman: Maksimal *{exposure}%* dari portofolio.\n"
            pesan_utama += f"• Stop Loss (95% Aman): {format_rupiah(var95_idr)}\n"
            pesan_utama += f"• Posisi Bandar (VWAP): {vwap_status}\n"
            pesan_utama += f"• Psikologi Pasar Dunia: {fng_str}\n\n"
            
            # BAGIAN BARU: SARAN EKSEKUSI
            pesan_utama += f"🎯 *KESIMPULAN & SARAN TINDAKAN*\n"
            pesan_utama += f"*{saran_tindakan}*\n"
            pesan_utama += f"_{alasan_saran}_"

            # ============================================================
            # PEMBUATAN PESAN DIAGNOSTIK (SMART ALERT: HANYA JIKA ERROR)
            # ============================================================
            pesan_diag = ""
            if ada_error_sistem:
                pesan_diag += f"⚠️ *PERINGATAN GANGGUAN SISTEM BOT* ⚠️\n\n"
                pesan_diag += "Bagian berikut mendeteksi masalah:\n"
                for k, v in diagnostics.items():
                    if "✅" not in v: 
                        pesan_diag += f"❌ *{k.upper()}*: {v}\n"
                pesan_diag += f"\n_Bot otomatis menggunakan fallback estimasi. Harap pantau server._"

            send_telegram_messages(pesan_utama, ["chart_main.png", "chart_zoom.png", "chart_indicators.png"], pesan_diag)
            print(f"[*] Pesan Telegram terkirim. Alasan: {alasan_kirim}")
        else:
            print(f"[*] SILENT MODE AKTIF: Harga stabil. Bot menunggu laporan terjadwal. Eksekusi: {time.time() - start_time:.1f} dtk.")

    except Exception as fatal_e:
        pesan_fatal = f"🚨 *BOT MATI MENDADAK (FATAL ERROR)* 🚨\n\nPenyebab: {str(fatal_e)}\nWaktu: {sekarang_wita.strftime('%H:%M WITA')}"
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_fatal})

if __name__ == "__main__":
    main()