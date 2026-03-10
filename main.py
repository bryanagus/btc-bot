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
import logging
from datetime import datetime

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
HISTORY_FILE = "ai_history_log.csv" 
WEB_DATA_FILE = "dashboard_data.json"
ALERT_FILE = "last_alert_state.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_system.log"),
        logging.StreamHandler()
    ]
)

diagnostics = {
    "api": "Normal",
    "csv": "Synchronized",
    "ai_1h": "Optimal",
    "ai_6h": "Optimal",
    "chart": "Optimal",
    "web3": "Updated"
}

HEADERS_BOT = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json'
}

def format_rupiah(angka):
    if pd.isna(angka): return "Rp 0"
    return f"Rp {angka:,.0f}".replace(',', '.')

def format_usd(angka):
    if pd.isna(angka): return "$0"
    return f"${angka:,.2f}"

def fetch_fear_and_greed():
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=100", headers=HEADERS_BOT, timeout=10).json()
        fng_dict = {pd.to_datetime(x['timestamp'], unit='s', utc=True).date(): float(x['value']) for x in resp['data']}
        return fng_dict
    except Exception as e:
        diagnostics["api"] = "Warning | F&G API Failed"
        logging.warning(f"F&G API Gagal: {e}")
        return {}

def fetch_data_with_retry(period='730d', interval='1h'):
    delays = [5, 15, 30]
    for attempt, delay in enumerate(delays + [0]):
        try:
            df = yf.download('BTC-USD', period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.droplevel(1)
            df.index = pd.to_datetime(df.index, utc=True)
            try:
                idr_data = yf.download('IDR=X', period='5d', progress=False)
                if isinstance(idr_data.columns, pd.MultiIndex): 
                    idr_data.columns = idr_data.columns.droplevel(1)
                kurs_idr = float(idr_data['Close'].dropna().iloc[-1])
            except Exception as e:
                kurs_idr = 15500.0
            try:
                indodax_req = requests.get('https://indodax.com/api/ticker/btcidr', headers=HEADERS_BOT, timeout=10).json()
                indodax_live_idr = float(indodax_req['ticker']['last'])
            except Exception as e:
                indodax_live_idr = float(df['Close'].iloc[-1]) * kurs_idr
                diagnostics["api"] = "Indodax API Down"
            return df, kurs_idr, indodax_live_idr
        except Exception as e:
            if attempt == len(delays):
                diagnostics["api"] = "Failed (API Global Down)"
                raise Exception(f"API Timeout: {e}")
            time.sleep(delay)

def fetch_indodax_depth(current_idr_price):
    try:
        resp = requests.get('https://indodax.com/api/depth/btcidr', headers=HEADERS_BOT, timeout=10).json()
        batas_atas = current_idr_price * 1.02
        batas_bawah = current_idr_price * 0.98
        buy_wall = sum([float(x[0]) * float(x[1]) for x in resp.get('buy', []) if float(x[0]) >= batas_bawah])
        sell_wall = sum([float(x[0]) * float(x[1]) for x in resp.get('sell', []) if float(x[0]) <= batas_atas])
        return buy_wall, sell_wall
    except Exception as e:
        diagnostics["api"] = "Warning | Depth API Failed"
        return 0, 0

def fetch_crypto_news_sentiment():
    try:
        rss_urls = ['https://www.coindesk.com/arc/outboundfeeds/rss/', 'https://cointelegraph.com/rss']
        analyzer = SentimentIntensityAnalyzer()
        crypto_lexicon = {"bullish": 2.5, "bearish": -2.5, "rekt": -3.0, "moon": 2.5, "pump": 2.0, "dump": -2.5, "hack": -3.0, "scam": -3.0}
        analyzer.lexicon.update(crypto_lexicon)
        compound_scores = []
        for url in rss_urls:
            response = requests.get(url, headers=HEADERS_BOT, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall('.//item')[:10]:
                    title = item.find('title').text
                    if title:
                        compound_scores.append(analyzer.polarity_scores(title)['compound'])
        if not compound_scores: return "NEUTRAL"
        avg = sum(compound_scores) / len(compound_scores)
        if avg >= 0.25: return "HIGHLY POSITIVE"
        elif avg <= -0.25: return "HIGHLY NEGATIVE"
        return "NEUTRAL"
    except Exception as e:
        return "UNAVAILABLE"

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
        df['Volume_Spike'] = df['Volume'] / df['Volume'].rolling(24).mean().replace(0, 1)
    else:
        df['VWAP_24'] = df['Close']
        df['Volume_Spike'] = 1.0
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
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["ATR_Ratio"] = df["ATR"] / df["Close"]
    df['Date_Only'] = df.index.date
    df['FnG_Index'] = df['Date_Only'].map(fng_dict)
    df['FnG_Index'] = df['FnG_Index'].ffill().bfill().fillna(50.0) 
    df.drop(columns=['Date_Only'], inplace=True)
    df["Target_1H"] = (df["Close"].shift(-1) > (df["Close"] * 1.007)).astype(float)
    df["Target_6H"] = (df["Close"].shift(-6) > (df["Close"] * 1.012)).astype(float)
    features_cols = [
        "EMA_Spread", "RSI", "MACD_Hist", "Log_Return", "Volatility", 
        "Trend_Slope", "Momentum_Accel", "Regime", "ADX", "BB_Width", "ATR_Ratio", "Volume_Spike"
    ]
    return df, features_cols

def train_honest_model(df, features, target_col, shift_len, model_name):
    model_file = f"ai_model_{model_name}.pkl"
    need_training = True
    X_live = df[features].iloc[-1:].fillna(0).values
    if os.path.exists(model_file):
        try:
            with open(model_file, 'rb') as f:
                models = pickle.load(f)
            last_trained = models.get('last_trained', 0)
            if (time.time() - last_trained) < 10800:
                need_training = False
                xgb_cal = models['xgb']
                mlp_cal = models['mlp']
                lr_cal = models['lr']
        except Exception as e:
            need_training = True
    if need_training:
        train_df = df.iloc[:-shift_len].dropna(subset=features + [target_col])
        X_train = train_df[features].values
        y_train = train_df[target_col].values
        tscv = TimeSeriesSplit(n_splits=5)
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('lr', LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'))
        ])
        lr_cal = CalibratedClassifierCV(lr_pipeline, method="sigmoid", cv=tscv)
        lr_cal.fit(X_train, y_train)
        mlp_pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True))
        ])
        mlp_cal = CalibratedClassifierCV(mlp_pipeline, method="sigmoid", cv=tscv)
        mlp_cal.fit(X_train, y_train)
        xgb_base = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, 
            eval_metric='logloss'
        )
        xgb_cal = CalibratedClassifierCV(xgb_base, method="sigmoid", cv=tscv)
        xgb_cal.fit(X_train, y_train)
        with open(model_file, 'wb') as f:
            pickle.dump({
                'lr': lr_cal, 
                'xgb': xgb_cal, 
                'mlp': mlp_cal, 
                'last_trained': time.time()
            }, f)
    prob_lr = lr_cal.predict_proba(X_live)[0, 1]
    prob_mlp = mlp_cal.predict_proba(X_live)[0, 1]
    prob_xgb = xgb_cal.predict_proba(X_live)[0, 1]
    prob_final = (prob_xgb * 0.50) + (prob_mlp * 0.30) + (prob_lr * 0.20)
    return prob_final

def get_historical_indodax_price(t_target_utc, past_usd, past_idr, actual_usd):
    try:
        t_unix = int(t_target_utc.timestamp())
        url = f"https://indodax.com/tradingview/history?symbol=BTCIDR&resolution=60&from={t_unix-3600}&to={t_unix+3600}"
        resp = requests.get(url, headers=HEADERS_BOT, timeout=10).json()
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
    waktu_eksekusi = datetime.now(pytz.utc)
    candle_time = df.index[-1] 
    usd_live_price = float(df['Close'].iloc[-1])
    target_time_1h = candle_time + pd.Timedelta(hours=1)
    target_time_6h = candle_time + pd.Timedelta(hours=6)
    eval_1h_msg = "Pending"
    eval_6h_msg = "Pending"
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
            mask_1h = (history['target_1h'] <= candle_time) & (history['result_1h'].isna())
            for idx, row in history[mask_1h].iterrows():
                t_target = row['target_1h']
                if t_target in df.index:
                    past_prob = float(row['prob_1h'])
                    past_usd_price = float(row['usd_start_price'])
                    past_idr_price = float(row['idr_start_price'])
                    actual_end_price = float(df.loc[t_target, 'Close'])
                    batas_naik_1h = past_usd_price * 1.007 
                    batas_turun_1h = past_usd_price * 0.993
                    if actual_end_price > batas_naik_1h: arah_asli = "Significant Uptrend"
                    elif actual_end_price > past_usd_price: arah_asli = "Slight Uptrend"
                    elif actual_end_price < batas_turun_1h: arah_asli = "Significant Downtrend"
                    else: arah_asli = "Slight Downtrend"
                    batas_target_ai_1h = past_usd_price * 1.007
                    if (past_prob > 0.5 and actual_end_price > batas_target_ai_1h) or (past_prob <= 0.5 and actual_end_price <= batas_target_ai_1h):
                        hasil = f"SUCCESS ({arah_asli})"
                    else:
                        hasil = f"FAILED ({arah_asli})"
                    history.loc[idx, 'result_1h'] = hasil
                    history.loc[idx, 'usd_end_price_1h'] = actual_end_price
                    history.loc[idx, 'idr_end_price_1h'] = get_historical_indodax_price(t_target, past_usd_price, past_idr_price, actual_end_price)
                    if t_target == candle_time: eval_1h_msg = hasil
                elif candle_time > t_target + pd.Timedelta(hours=2):
                    history.loc[idx, 'result_1h'] = "DATA MISSING"
            mask_6h = (history['target_6h'] <= candle_time) & (history['result_6h'].isna())
            for idx, row in history[mask_6h].iterrows():
                t_target = row['target_6h']
                if t_target in df.index:
                    past_prob = float(row['prob_6h'])
                    past_usd_price = float(row['usd_start_price'])
                    past_idr_price = float(row['idr_start_price'])
                    actual_end_price = float(df.loc[t_target, 'Close'])
                    batas_naik_6h = past_usd_price * 1.012 
                    batas_turun_6h = past_usd_price * 0.988
                    if actual_end_price > batas_naik_6h: arah_asli = "Significant Uptrend"
                    elif actual_end_price > past_usd_price: arah_asli = "Slight Uptrend"
                    elif actual_end_price < batas_turun_6h: arah_asli = "Significant Downtrend"
                    else: arah_asli = "Slight Downtrend"
                    batas_target_ai_6h = past_usd_price * 1.012
                    if (past_prob > 0.5 and actual_end_price > batas_target_ai_6h) or (past_prob <= 0.5 and actual_end_price <= batas_target_ai_6h):
                        hasil = f"SUCCESS ({arah_asli})"
                    else:
                        hasil = f"FAILED ({arah_asli})"
                    history.loc[idx, 'result_6h'] = hasil
                    history.loc[idx, 'usd_end_price_6h'] = actual_end_price
                    history.loc[idx, 'idr_end_price_6h'] = get_historical_indodax_price(t_target, past_usd_price, past_idr_price, actual_end_price)
                    if t_target == candle_time: eval_6h_msg = hasil
                elif candle_time > t_target + pd.Timedelta(hours=2):
                    history.loc[idx, 'result_6h'] = "DATA MISSING"
            history.to_csv(HISTORY_FILE, index=False)
            recent_1h = history.dropna(subset=['result_1h']).tail(24)
            recent_valid = recent_1h[~recent_1h['result_1h'].astype(str).str.contains("MISSING")]
            if len(recent_valid) >= 3:
                benar_count = recent_valid['result_1h'].astype(str).str.contains('SUCCESS').sum()
                win_rate = benar_count / len(recent_valid)
                if win_rate < 0.4: risk_multiplier = 0.1     
                elif win_rate < 0.6: risk_multiplier = 0.5   
                else: risk_multiplier = 1.0                  
    except Exception as e:
        diagnostics["csv"] = f"Error: {e}"
        risk_multiplier = 1.0
    try:
        t_curr_str = waktu_eksekusi.strftime('%Y-%m-%dT%H:%M:%SZ')
        t_1h_str = target_time_1h.strftime('%Y-%m-%dT%H:%M:%SZ')
        t_6h_str = target_time_6h.strftime('%Y-%m-%dT%H:%M:%SZ')
        is_duplicate = False
        if file_exists:
            try:
                if len(history) > 0 and str(history['created_at'].iloc[-1])[:16] == t_curr_str[:16]:
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
        diagnostics["csv"] = "Failed Writing CSV"
    return eval_1h_msg, eval_6h_msg, risk_multiplier

def generate_web3_dashboard_data(indodax_idr, global_usd, kurs_idr, prob_1h, prob_6h, df, risk_mult):
    global_idr_converted = global_usd * kurs_idr
    spread_premium = indodax_idr - global_idr_converted
    threshold_1h = 0.08
    threshold_6h = 0.15
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
                "prediksi": "UPTREND" if row['prob_1h'] > threshold_1h else "DOWNTREND",
                "keyakinan": f"{row['prob_1h']*100:.1f}%",
                "status": str(row['result_1h']).split(' ')[0]
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
                "prediksi": "UPTREND" if row['prob_6h'] > threshold_6h else "DOWNTREND",
                "keyakinan": f"{row['prob_6h']*100:.1f}%",
                "status": str(row['result_6h']).split(' ')[0]
            })
        total_1h = len([x for x in table_1h if "SUCCESS" in x['status']])
        total_6h = len([x for x in table_6h if "SUCCESS" in x['status']])
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
                "arah_1h": "UPTREND" if prob_1h > threshold_1h else "DOWNTREND",
                "arah_6h": "UPTREND" if prob_6h > threshold_6h else "DOWNTREND"
            },
            "stats": {
                "win_rate_1h": round((total_1h / len(table_1h) * 100) if table_1h else 0, 1),
                "win_rate_6h": round((total_6h / len(table_6h) * 100) if table_6h else 0, 1),
                "risk_multiplier": risk_mult
            },
            "indicators": {
                "rsi": round(df['RSI'].iloc[-1], 2),
                "adx": round(df['ADX'].iloc[-1], 2),
                "trend": "UPTREND" if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else "DOWNTREND"
            },
            "history_1h": table_1h[::-1],
            "history_6h": table_6h[::-1]
        }
        with open(WEB_DATA_FILE, 'w') as f:
            json.dump(web_data, f, indent=4)
    except Exception as e:
        diagnostics["web3"] = f"JSON Export Failed: {e}"

def calculate_var_95(price, log_returns_series):
    var_log_return = np.percentile(log_returns_series.dropna(), 5)
    var_price = price * np.exp(var_log_return)
    return var_price

def position_sizing_kelly(prob_1h, prob_6h, atr, risk_multiplier):
    threshold_avg = (0.08 + 0.15) / 2
    avg_prob = (prob_1h + prob_6h) / 2
    if avg_prob < threshold_avg: return 0.0
    reward, risk = atr * 3.5, atr * 1.5
    if risk <= 0: return 0.0 
    norm_prob = 0.5 + (avg_prob - threshold_avg) 
    edge = norm_prob - ((1 - norm_prob) / (reward/risk))
    size = min(max(edge, 0), 0.25) * risk_multiplier 
    return round(size * 100, 2)

def plot_professional_analysis(df, prob_1h, prob_6h, atr, base_filename="chart_main"):
    plt.clf() 
    plot_data = df.tail(80).copy() 
    plot_data.index = plot_data.index.tz_convert('Asia/Makassar')
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 8))
    plt.plot(plot_data.index, plot_data['Close'], label='Global Price (USD)', color='black', linewidth=2, zorder=5)
    plt.plot(plot_data.index, plot_data['VWAP_24'], label='VWAP', color='#ff7f0e', linestyle='-.')
    last_time = plot_data.index[-1]
    last_price = plot_data['Close'].iloc[-1]
    t_1h = last_time + pd.Timedelta(hours=1)
    p_1h = last_price + (atr * (prob_1h - 0.5) * 2)
    t_6h = last_time + pd.Timedelta(hours=6)
    p_6h = last_price + (atr * (prob_6h - 0.5) * 4)
    plt.scatter(t_1h, p_1h, color='blue', s=80, marker='o', zorder=10, label='1H Target')
    plt.plot([last_time, t_1h], [last_price, p_1h], color='blue', linestyle='--', linewidth=2)
    plt.scatter(t_6h, p_6h, color='red', s=100, marker='X', zorder=10, label='6H Target')
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
    ax1.set_title('ZOOM BOLLINGER (USD GLOBAL)', fontsize=16, fontweight='bold')
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
        ax1.set_title('1. PRICE TREND & EMA', fontweight='bold')
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
        diagnostics["chart"] = f"Render Failed: {e}"

def send_telegram_messages(pesan_utama, chart_paths, pesan_diag="", pesan_saran=""):
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
    if pesan_saran.strip() != "":
        requests.post(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_saran, 'parse_mode': 'Markdown'})

def main():
    sekarang_wita = datetime.now(pytz.timezone('Asia/Makassar'))
    try:
        fng_dict = fetch_fear_and_greed()
        df, kurs_idr, indodax_live_idr = fetch_data_with_retry()
        df, features = engineer_features(df, fng_dict)
        df_closed = df.iloc[:-1].copy()
        buy_wall, sell_wall = fetch_indodax_depth(indodax_live_idr)
        status_berita = fetch_crypto_news_sentiment()
        global_usd = float(df['Close'].iloc[-1])
        global_idr_converted = global_usd * kurs_idr
        spread_premium = indodax_live_idr - global_idr_converted
        current_atr = df["ATR"].iloc[-1]
        try: prob_1h = train_honest_model(df_closed, features, "Target_1H", shift_len=1, model_name="1H")
        except Exception as e: 
            diagnostics["ai_1h"] = f"Error: {e}"
            prob_1h = 0.5
        try: prob_6h = train_honest_model(df_closed, features, "Target_6H", shift_len=6, model_name="6H")
        except Exception as e: 
            diagnostics["ai_6h"] = f"Error: {e}"
            prob_6h = 0.5
        eval_1h, eval_6h, risk_mult = manage_history_and_evaluate(df, prob_1h, prob_6h, indodax_live_idr, kurs_idr)
        generate_web3_dashboard_data(indodax_live_idr, global_usd, kurs_idr, prob_1h, prob_6h, df, risk_mult)
        exposure = position_sizing_kelly(prob_1h, prob_6h, current_atr, risk_mult)
        var95_usd = calculate_var_95(global_usd, df["Log_Return"]) 
        var95_idr = var95_usd * kurs_idr
        threshold_1h = 0.08
        threshold_6h = 0.15
        def get_arah(prob, threshold):
            if prob >= (threshold * 1.5): return "Strong Uptrend"
            elif prob > threshold: return "Uptrend"
            elif prob <= (threshold * 0.5): return "Strong Downtrend"
            else: return "Downtrend"
        arah_1h = get_arah(prob_1h, threshold_1h)
        arah_6h = get_arah(prob_6h, threshold_6h)
        fng_val = df['FnG_Index'].iloc[-1]
        saran_tindakan = ""
        alasan_saran = ""
        rsi_sekarang = df['RSI'].iloc[-1]
        if prob_1h > threshold_1h and prob_6h > threshold_6h and spread_premium <= 0:
            if status_berita == "HIGHLY NEGATIVE":
                saran_tindakan = "HOLD (High Headline Risk)"
                alasan_saran = "Teknikal menunjukkan uptrend, namun sentimen makro/berita sangat negatif. Hindari eksekusi untuk meminimalisir volatilitas tak terduga."
            elif sell_wall > (buy_wall * 3):
                saran_tindakan = "HOLD (Liquidity Wall Detected)"
                alasan_saran = "AI mendeteksi potensi kenaikan, namun terdapat resistensi likuiditas tinggi (Sell Wall) di bursa lokal."
            elif rsi_sekarang < 35:
                saran_tindakan = "WAIT (Oversold / Falling Knife)"
                alasan_saran = "Aset diperdagangkan pada valuasi diskon, namun momentum makro masih menunjukkan tekanan jual tinggi. Tunggu konfirmasi reversal."
            else:
                saran_tindakan = "STRONG BUY"
                alasan_saran = "Konfluensi sinyal positif: Algoritma AI mengindikasikan uptrend, aset berada di valuasi diskon (premium negatif), dan dukungan likuiditas stabil."
        elif prob_1h > threshold_1h and prob_6h > threshold_6h and spread_premium > (global_idr_converted * 0.005):
            if rsi_sekarang > 70:
                saran_tindakan = "TAKE PROFIT (Overbought)"
                alasan_saran = "Aset berada di zona jenuh beli (RSI > 70) dengan premium lokal yang tinggi. Direkomendasikan untuk merealisasikan keuntungan."
            else:
                saran_tindakan = "PARTIAL TAKE PROFIT"
                alasan_saran = "Tren global positif, namun harga aset di bursa lokal diperdagangkan dengan premium yang tidak proporsional. Kurangi eksposur secara bertahap."
        elif prob_1h <= threshold_1h and prob_6h <= threshold_6h and spread_premium > 0:
            saran_tindakan = "STRONG SELL"
            alasan_saran = "AI memproyeksikan koreksi harga global. Harga lokal saat ini memiliki premium positif, memberikan spread ideal untuk likuidasi aset."
        elif prob_1h <= threshold_1h and prob_6h <= threshold_6h and spread_premium < 0:
            saran_tindakan = "HOLD (Await Bottom)"
            alasan_saran = "Proyeksi makro masih menunjukkan downtrend. Pertahankan posisi kas (cash) hingga algoritma mendeteksi titik support absolut."
        else:
            saran_tindakan = "NEUTRAL"
            alasan_saran = "Divergensi sinyal antar timeframe. Volatilitas diproyeksikan stagnan. Tidak direkomendasikan untuk membuka posisi baru."
        if fng_val >= 75: fng_str = f"{int(fng_val)} (Extreme Greed / Overextended)"
        elif fng_val >= 55: fng_str = f"{int(fng_val)} (Greed / Bullish Bias)"
        elif fng_val <= 25: fng_str = f"{int(fng_val)} (Extreme Fear / Undervalued)"
        elif fng_val <= 45: fng_str = f"{int(fng_val)} (Fear / Bearish Bias)"
        else: fng_str = f"{int(fng_val)} (Neutral)"
        if buy_wall > sell_wall:
            status_orderbook = f"Demand Dominant ({(buy_wall/sell_wall if sell_wall>0 else 1):.1f}x Bid/Ask Ratio)"
        else:
            status_orderbook = f"Supply Dominant ({(sell_wall/buy_wall if buy_wall>0 else 1):.1f}x Ask/Bid Ratio)"
        ada_error_sistem = any("Normal" not in v and "Synchronized" not in v and "Optimal" not in v and "Updated" not in v for v in diagnostics.values())
        kirim_telegram = False
        alasan_kirim = ""
        is_routine_update = False
        teks_perubahan = "Stabil 0.00%" 
        waktu_sekarang_str = sekarang_wita.strftime('%Y-%m-%d-%H')
        menit_sekarang = sekarang_wita.minute
        last_state = {}
        if not os.path.exists(ALERT_FILE):
            kirim_telegram = True
            alasan_kirim = "System Initialization / Reboot"
            is_routine_update = True
        elif ada_error_sistem:
            kirim_telegram = True
            alasan_kirim = "System/API Disruption Detected"
        else:
            try:
                with open(ALERT_FILE, 'r') as f:
                    last_state = json.load(f)
                last_price_saved = last_state.get('price', global_usd)
                if last_price_saved > 0:
                    raw_diff_pct = ((global_usd - last_price_saved) / last_price_saved) * 100
                else:
                    raw_diff_pct = 0.0
                if raw_diff_pct > 0: teks_perubahan = f"Naik {raw_diff_pct:.2f}%"
                elif raw_diff_pct < 0: teks_perubahan = f"Turun {abs(raw_diff_pct):.2f}%"
                else: teks_perubahan = f"Stabil 0.00%"
                price_diff = abs(raw_diff_pct)
                selisih_waktu = time.time() - last_state.get('time', 0)
                last_hourly_report = last_state.get('last_hourly_report', '')
                if price_diff >= 0.6:
                    kirim_telegram = True
                    alasan_kirim = f"Significant Price Movement ({teks_perubahan})"
                elif saran_tindakan in ["STRONG BUY", "STRONG SELL", "WAIT (Oversold / Falling Knife)", "TAKE PROFIT (Overbought)", "PARTIAL TAKE PROFIT"] and saran_tindakan != last_state.get('saran', ''):
                    kirim_telegram = True
                    alasan_kirim = f"CRITICAL SIGNAL CHANGE"
                elif saran_tindakan != last_state.get('saran', ''):
                    if selisih_waktu < 3600: kirim_telegram = False 
                    else:
                        kirim_telegram = True
                        alasan_kirim = f"Recommendation Update"
                elif 5 <= menit_sekarang < 20 and last_hourly_report != waktu_sekarang_str:
                    kirim_telegram = True
                    alasan_kirim = f"Routine Market Update ({sekarang_wita.strftime('%H:10')} WITA)"
                    is_routine_update = True
                else:
                    kirim_telegram = False
            except Exception as e:
                kirim_telegram = True
                alasan_kirim = f"System State Reset"
                is_routine_update = True
        plot_professional_analysis(df, prob_1h, prob_6h, current_atr, "chart_main")
        plot_zoomed_analysis(df, prob_1h, prob_6h, current_atr, "chart_zoom")
        plot_dashboard_indicators(df, "chart_indicators")
        if kirim_telegram:
            new_last_hourly = waktu_sekarang_str if is_routine_update else last_state.get('last_hourly_report', '')
            with open(ALERT_FILE, 'w') as f:
                json.dump({
                    'price': global_usd,
                    'signal': saran_tindakan, 
                    'saran': saran_tindakan, 
                    'time': time.time(),
                    'last_hourly_report': new_last_hourly
                }, f)
            pesan_utama = f"*QUANTITATIVE MARKET REPORT | BTC-IDR*\n"
            pesan_utama += f"Timestamp: {sekarang_wita.strftime('%Y-%m-%d %H:%M WITA')}\n"
            pesan_utama += f"Trigger: {alasan_kirim}\n"
            pesan_utama += f"Movement: {teks_perubahan} (Since last report)\n\n"
            pesan_utama += f"*MARKET DATA & ARBITRAGE*\n"
            pesan_utama += f"• Global Spot (USD): {format_usd(global_usd)}\n"
            pesan_utama += f"• FX Rate (USD/IDR): {format_rupiah(kurs_idr)}\n"
            pesan_utama += f"• Fair Value (IDR): *{format_rupiah(global_idr_converted)}*\n"
            pesan_utama += f"• Indodax Spot: *{format_rupiah(indodax_live_idr)}*\n"
            if spread_premium > 0:
                pesan_utama += f"• Premium/Discount: +{format_rupiah(abs(spread_premium))} (Premium)\n"
            else:
                pesan_utama += f"• Premium/Discount: -{format_rupiah(abs(spread_premium))} (Discount)\n"
            pesan_utama += f"• Liquidity Profile: {status_orderbook}\n\n"
            pesan_utama += f"*ALGORITHMIC PROJECTION (ENSEMBLE MODEL)*\n"
            pesan_utama += f"*Short-Term (1H)*\n"
            pesan_utama += f"• Direction: *{arah_1h}* (Confidence: {prob_1h*100:.1f}%)\n"
            pesan_utama += f"• Previous 1H Outcome: {eval_1h}\n\n"
            pesan_utama += f"*Mid-Term (6H)*\n"
            pesan_utama += f"• Direction: *{arah_6h}* (Confidence: {prob_6h*100:.1f}%)\n"
            pesan_utama += f"• Previous 6H Outcome: {eval_6h}\n\n"
            pesan_utama += f"*RISK METRICS & SENTIMENT*\n"
            if risk_mult < 1.0:
                pesan_utama += f"*SYSTEM ALERT:* Auto-deleveraging active due to recent model variance.\n"
            pesan_utama += f"• Recommended Exposure: *{exposure}%* of portfolio.\n"
            pesan_utama += f"• VaR (95% Confidence): {format_rupiah(var95_idr)}\n"
            pesan_utama += f"• Global Market Sentiment: {fng_str}\n\n"
            pesan_diag = ""
            if ada_error_sistem:
                pesan_diag += f"*SYSTEM ALERT*\n\n"
                for k, v in diagnostics.items():
                    if "Normal" not in v and "Synchronized" not in v and "Optimal" not in v and "Updated" not in v:
                        pesan_diag += f"- *{k.upper()}*: {v}\n"
                pesan_diag += f"\n_Fallback estimation active._"
            pesan_saran = f"*ACTIONABLE INSIGHT*\n"
            pesan_saran += f"Status: *{saran_tindakan}*\n"
            pesan_saran += f"Rationale: {alasan_saran}"
            send_telegram_messages(pesan_utama, ["chart_main.png", "chart_zoom.png", "chart_indicators.png"], pesan_diag, pesan_saran)
    except Exception as fatal_e:
        pesan_fatal = f"*SYSTEM FAILURE*\n\nCause: {str(fatal_e)}\nTime: {sekarang_wita.strftime('%H:%M WITA')}"
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_fatal})

if __name__ == "__main__":
    main()
