import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs'))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import shutil
import warnings
import xml.etree.ElementTree as ET
import time
import csv
import json
import pickle
import logging
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import pytesseract
from PIL import Image
import io
import re

# ==============================================================================
# ⚙️ PANEL KONFIGURASI BOT ULTIMATE (GITHUB ACTIONS READY)
# ==============================================================================

# 1. KREDENSIAL API (Diambil otomatis dari GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'MASUKKAN_TOKEN_BOT_TELEGRAM_ANDA')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'MASUKKAN_CHAT_ID_ANDA')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'MASUKKAN_API_KEY_GEMINI_ANDA')

# 2. KONFIGURASI TRADING DASAR
FEE_MAKER_TOTAL = 0.4322           # Total Fee Indodax (Beli + Jual)
MIN_TRADE_IDR = 25000              # Minimal transaksi limit/maker Indodax
STOP_LOSS_NET = -15.0              # Batas Cut Loss PnL bersih (%)

# 3. KONFIGURASI SMART LOCK-IN (TAKE PROFIT)
BREAK_EVEN_TRIGGER_PCT = 2.5       # Target % naik untuk mengunci Break-Even
BREAK_EVEN_LOCK_PCT = 0.2          # Target sisa % aman saat harga jatuh
TRAILING_PROFIT_DROP_PCT = 1.2     # Toleransi % harga turun dari pucuk sebelum take profit

# 4. KONFIGURASI MANAJEMEN PELURU (DYNAMIC SIZING & ADAPTIVE SNIPER)
BASE_RISK_PCT = 0.50               # Menggunakan 50% dari sisa saldo tunai
MAX_RISK_PCT = 0.80                # Menggunakan 80% dari sisa saldo tunai jika AI sangat yakin
# Catatan: Jarak Drop dan Bounce dihitung otomatis oleh mesin berdasarkan volatilitas pasar

# 5. KONFIGURASI BACKTEST
BACKTEST_INITIAL_CAPITAL = 50000   
BACKTEST_TEST_SIZE = 0.3           

# ==============================================================================
# 🛑 ENGINE UTAMA - JANGAN DIUBAH
# ==============================================================================

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
pd.options.mode.copy_on_write = True

HISTORY_FILE = "ai_history_log.csv"
WEB_DATA_FILE = "dashboard_data.json"
ALERT_FILE = "last_alert_state.json"
OHLCV_FILE = "indodax_ohlcv.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("bot_system.log"), logging.StreamHandler()])
diagnostics = {"api": "Normal", "csv": "Synchronized", "ai_1h": "Optimal", "ai_6h": "Optimal", "chart": "Optimal", "web3": "Updated", "macro": "Updated"}
HEADERS_BOT = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36', 'Accept': 'application/json'}

def get_requests_session():
    session = requests.Session()
    retry = Retry(total=4, read=4, connect=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

http_session = get_requests_session()

def check_user_telegram_updates(last_update_id):
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "MASUKKAN_TOKEN_BOT_TELEGRAM_ANDA":
        return None, 0.0, last_update_id
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {'offset': last_update_id + 1, 'timeout': 5}
    aksi_user = None
    harga_input = 0.0
    new_last_id = last_update_id
    try:
        response = http_session.get(url, params=params, timeout=10).json()
        if response.get('ok') and response.get('result'):
            for update in response['result']:
                new_last_id = update['update_id']
                message = update.get('message', {})
                teks = message.get('text', message.get('caption', '')).lower()
                if "beli " in teks:
                    try:
                        angka_str = ''.join(filter(str.isdigit, teks))
                        if angka_str and float(angka_str) > 100000000:
                            harga_input = float(angka_str)
                            aksi_user = "BUY"
                    except: pass
                elif "jual" in teks or "kosong" in teks:
                    aksi_user = "SELL"
                    harga_input = 0.0
                elif 'photo' in message:
                    try:
                        file_id = message['photo'][-1]['file_id']
                        file_info = http_session.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}").json()
                        file_path = file_info['result']['file_path']
                        img_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
                        img_data = http_session.get(img_url).content
                        img = Image.open(io.BytesIO(img_data))
                        teks_ocr = pytesseract.image_to_string(img, lang='ind+eng').lower()
                        if "berhasil" in teks_ocr or "selesai" in teks_ocr or "beli" in teks_ocr:
                            angka_kotor = re.sub(r'[^0-9]', '', teks_ocr)
                            match = re.search(r'(1\d{8,9})', angka_kotor)
                            if match:
                                harga_input = float(match.group(1))
                                aksi_user = "BUY"
                        elif "jual" in teks_ocr:
                            aksi_user = "SELL"
                            harga_input = 0.0
                    except Exception as e:
                        print(f"Gagal OCR: {e}")
    except Exception as e:
        print(f"Gagal getUpdates: {e}")
    return aksi_user, harga_input, new_last_id

def format_rupiah(angka):
    if pd.isna(angka) or angka is None: return "Rp 0"
    return f"Rp {int(angka):,.0f}".replace(',', '.')

def fetch_fear_and_greed():
    try:
        resp = http_session.get("https://api.alternative.me/fng/?limit=100", headers=HEADERS_BOT, timeout=10).json()
        fng_dict = {pd.to_datetime(x['timestamp'], unit='s', utc=True).date(): float(x['value']) for x in resp['data']}
        return fng_dict
    except Exception as e:
        diagnostics["api"] = "Warning"
        return {}

def fetch_indodax_usdt_kurs():
    try:
        resp = http_session.get('https://indodax.com/api/ticker/usdtidr', headers=HEADERS_BOT, timeout=10).json()
        return float(resp['ticker']['last'])
    except Exception as e:
        if os.path.exists(HISTORY_FILE):
            try:
                hist = pd.read_csv(HISTORY_FILE)
                if 'kurs_usd_idr' in hist.columns and not hist.empty:
                    last_kurs = float(hist['kurs_usd_idr'].dropna().iloc[-1])
                    if last_kurs > 10000:
                        return last_kurs
            except: pass
        return 15500.0

def fetch_data_incremental():
    try:
        if os.path.exists(OHLCV_FILE):
            df_old = pd.read_csv(OHLCV_FILE, index_col='Time', parse_dates=True)
            if df_old.index.tz is None: df_old.index = df_old.index.tz_localize('UTC')
            else: df_old.index = df_old.index.tz_convert('UTC')
            last_time = int(df_old.index[-1].timestamp()) - 3600
        else:
            df_old = pd.DataFrame()
            last_time = int(time.time()) - (365 * 24 * 60 * 60)
        end_time = int(time.time())
        chunk_size = 15 * 24 * 60 * 60
        all_data = []
        current_start = last_time
        while current_start < end_time:
            current_end = min(current_start + chunk_size, end_time)
            url = f"https://indodax.com/tradingview/history_v2?symbol=BTCIDR&tf=60&from={current_start}&to={current_end}"
            for attempt in range(3):
                try:
                    resp = http_session.get(url, headers=HEADERS_BOT, timeout=10).json()
                    if isinstance(resp, list):
                        all_data.extend(resp)
                        break
                except: time.sleep(2)
            current_start = current_end + 1
            time.sleep(0.5)
        if all_data:
            new_df = pd.DataFrame(all_data)
            new_df['Time'] = pd.to_datetime(new_df['Time'], unit='s', utc=True)
            new_df.set_index('Time', inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                new_df[col] = new_df[col].astype(float)
            df_combined = pd.concat([df_old, new_df])
        else:
            df_combined = df_old
        if df_combined.empty: raise Exception("Empty Data")
        df_combined.sort_index(inplace=True)
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        if len(df_combined) > 8000: df_combined = df_combined.tail(8000)
        df_combined.to_csv(OHLCV_FILE)
        indodax_live_idr = float(df_combined['Close'].iloc[-1])
        kurs_idr = fetch_indodax_usdt_kurs()
        global_usd = indodax_live_idr / kurs_idr
        return df_combined, kurs_idr, global_usd, indodax_live_idr
    except Exception as e:
        raise Exception(f"Fetch Error: {e}")

def fetch_indodax_depth(current_idr_price):
    try:
        resp = http_session.get('https://indodax.com/api/depth/btcidr', headers=HEADERS_BOT, timeout=10).json()
        batas_atas = current_idr_price * 1.02
        batas_bawah = current_idr_price * 0.98
        buy_wall = sum([float(x[0]) * float(x[1]) for x in resp.get('buy', []) if float(x[0]) >= batas_bawah])
        sell_wall = sum([float(x[0]) * float(x[1]) for x in resp.get('sell', []) if float(x[0]) <= batas_atas])
        return buy_wall, sell_wall
    except:
        diagnostics["api"] = "Warning"
        return 0, 0

def fetch_crypto_news_sentiment():
    try:
        rss_urls = ['https://www.coindesk.com/arc/outboundfeeds/rss/', 'https://cointelegraph.com/rss']
        analyzer = SentimentIntensityAnalyzer()
        crypto_lexicon = {"bullish": 3.0, "bearish": -3.0, "rekt": -4.0, "moon": 3.0, "pump": 2.5, "dump": -3.0, "hack": -4.0, "scam": -4.0, "etf": 2.0, "fud": -3.0, "liquidated": -3.0, "ath": 3.0, "approval": 2.0}
        analyzer.lexicon.update(crypto_lexicon)
        compound_scores = []
        headlines = []
        for url in rss_urls:
            try:
                response = http_session.get(url, headers=HEADERS_BOT, timeout=10)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    berita_ditemukan = 0
                    for item in root.findall('.//item'):
                        title = item.find('title').text
                        if title:
                            title_lower = title.lower()
                            kata_kunci = ['btc', 'bitcoin', 'crypto', 'market', 'sec', 'etf', 'fed', 'hack', 'binance', 'indodax', 'inflation', 'rate']
                            if any(k in title_lower for k in kata_kunci):
                                compound_scores.append(analyzer.polarity_scores(title)['compound'])
                                headlines.append(title)
                                berita_ditemukan += 1
                        if berita_ditemukan >= 5: break
            except: continue
        if not compound_scores: return "NEUTRAL", []
        avg = sum(compound_scores) / len(compound_scores)
        if avg >= 0.20: status = "HIGHLY POSITIVE"
        elif avg <= -0.20: status = "HIGHLY NEGATIVE"
        else: status = "NEUTRAL"
        return status, headlines
    except:
        return "UNAVAILABLE", []

def fetch_global_macro_data():
    macro_data = {"DXY": 0.0, "SP500": 0.0, "status": "UNAVAILABLE"}
    headers_macro = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 'Accept': 'application/json'}
    try:
        r_dxy = http_session.get("https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?interval=1d&range=1d", headers=headers_macro, timeout=5)
        if r_dxy.status_code == 200:
            macro_data["DXY"] = round(r_dxy.json()['chart']['result'][0]['meta']['regularMarketPrice'], 2)
        r_spy = http_session.get("https://query1.finance.yahoo.com/v8/finance/chart/^GSPC?interval=1d&range=1d", headers=headers_macro, timeout=5)
        if r_spy.status_code == 200:
            macro_data["SP500"] = round(r_spy.json()['chart']['result'][0]['meta']['regularMarketPrice'], 2)
        if macro_data["DXY"] > 0:
            macro_data["status"] = "UPDATED"
    except Exception as e:
        diagnostics["macro"] = "Failed"
    return macro_data

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
    df['ATR_Ratio'] = df["ATR"] / df["Close"]
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['ADX'] = (100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']).replace(0,1)).ewm(alpha=1/14, adjust=False).mean()
    df['VWAP_24'] = (((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']).rolling(24).sum() / df['Volume'].rolling(24).sum()
    df['Volume_Spike'] = df['Volume'] / df['Volume'].rolling(24).mean().replace(0, 1)
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
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["Rolling_Max_50"] = df["High"].rolling(50).max()
    df["Rolling_Min_50"] = df["Low"].rolling(50).min()
    df["Dist_to_Res"] = (df["Rolling_Max_50"] - df["Close"]) / df["Close"]
    df["Dist_to_Sup"] = (df["Close"] - df["Rolling_Min_50"]) / df["Close"]
    df["Swing_High"] = (df["High"] == df["High"].rolling(window=5, center=True).max()).astype(int)
    df["Swing_Low"] = (df["Low"] == df["Low"].rolling(window=5, center=True).min()).astype(int)
    df["Regime_Str"] = "RANGE"
    df.loc[df["ATR_Ratio"] > 0.006, "Regime_Str"] = "TREND"
    df.loc[df["Volatility"] > 0.01, "Regime_Str"] = "VOLATILE"
    df["Regime"] = np.where(df["EMA20"] > df["EMA50"], 1, 0)
    df['Date_Only'] = df.index.date
    df['FnG_Index'] = df['Date_Only'].map(fng_dict)
    df['FnG_Index'] = df['FnG_Index'].ffill().bfill().fillna(50.0)
    df.drop(columns=['Date_Only'], inplace=True)
    df["Target_1H"] = (df["Close"].shift(-1) > (df["Close"] * 1.002)).astype(float)
    df["Target_6H"] = (df["Close"].shift(-6) > (df["Close"] * 1.005)).astype(float)
    return df

def build_lstm_data(data_scaled, target, seq_len):
    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i+seq_len])
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)

def train_honest_model(df, target_col, shift_len, model_name, current_regime):
    features_xgb = ["EMA_Spread", "MACD_Hist", "Trend_Slope", "Momentum_Accel", "ADX"]
    features_mlp = ["Volatility", "ATR_Ratio", "BB_Width", "Log_Return"]
    features_lr = ["EMA_Spread", "BB_Width", "Trend_Slope", "Log_Return", "RSI"]
    features_rf = ["Volume_Spike", "BB_Width", "ATR_Ratio", "Log_Return", "RSI"]
    features_svm = ["Regime", "EMA_Spread", "Volatility", "Trend_Slope", "MACD_Hist"]
    features_lstm = ["Log_Return", "RSI", "MACD_Hist", "Volatility", "Volume_Spike"]
    model_file_pkl = f"ai_model_v3_{model_name}_{current_regime}.pkl"
    model_file_lstm = f"ai_model_v3_{model_name}_{current_regime}_lstm.keras"
    need_training = True
    if os.path.exists(model_file_pkl) and os.path.exists(model_file_lstm):
        try:
            with open(model_file_pkl, 'rb') as f:
                models = pickle.load(f)
            last_trained = models.get('last_trained', 0)
            if (time.time() - last_trained) < 10800:
                need_training = False
                xgb_model = models['xgb']
                mlp_cal = models['mlp']
                lr_cal = models['lr']
                rf_cal = models['rf']
                svm_cal = models['svm']
                scaler_lstm = models['scaler_lstm']
                lstm_model = load_model(model_file_lstm)
        except Exception as e:
            need_training = True
    if need_training:
        all_features = list(set(features_xgb + features_mlp + features_lr + features_rf + features_svm + features_lstm))
        train_df = df.iloc[:-shift_len].dropna(subset=all_features + [target_col])
        regime_df = train_df[train_df["Regime_Str"] == current_regime]
        if len(regime_df) < 100:
            regime_df = train_df
        y_train = np.ascontiguousarray(regime_df[target_col].values, dtype=np.int64)
        tscv = TimeSeriesSplit(n_splits=5)
        xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, subsample=0.6, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1)
        xgb_model.fit(np.ascontiguousarray(regime_df[features_xgb].values, dtype=np.float64), y_train)
        mlp_pipeline = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(16, 8), alpha=0.01, max_iter=300, random_state=42))])
        mlp_cal = CalibratedClassifierCV(mlp_pipeline, method="sigmoid", cv=tscv)
        mlp_cal.fit(np.ascontiguousarray(regime_df[features_mlp].values, dtype=np.float64), y_train)
        lr_pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=300, C=0.5, random_state=42, class_weight='balanced'))])
        lr_cal = CalibratedClassifierCV(lr_pipeline, method="sigmoid", cv=tscv)
        lr_cal.fit(np.ascontiguousarray(regime_df[features_lr].values, dtype=np.float64), y_train)
        rf_pipeline = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42, class_weight='balanced', n_jobs=-1))])
        rf_cal = CalibratedClassifierCV(rf_pipeline, method="sigmoid", cv=tscv)
        rf_cal.fit(np.ascontiguousarray(regime_df[features_rf].values, dtype=np.float64), y_train)
        svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, C=0.5, random_state=42, class_weight='balanced'))])
        svm_cal = CalibratedClassifierCV(svm_pipeline, method="sigmoid", cv=tscv)
        svm_cal.fit(np.ascontiguousarray(regime_df[features_svm].values, dtype=np.float64), y_train)
        split_idx = int(len(regime_df) * 0.8)
        scaler_lstm = StandardScaler()
        scaler_lstm.fit(regime_df[features_lstm].values[:split_idx]) 
        X_lstm_scaled = scaler_lstm.transform(regime_df[features_lstm].values) 
        X_seq, y_seq = build_lstm_data(X_lstm_scaled, y_train, 5)
        if len(X_seq) > 10:
            lstm_model = Sequential()
            lstm_model.add(Input(shape=(5, len(features_lstm))))
            lstm_model.add(LSTM(8, activation='relu'))
            lstm_model.add(Dropout(0.4))
            lstm_model.add(Dense(1, activation='sigmoid'))
            lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            lstm_model.fit(X_seq, y_seq, epochs=15, batch_size=16, verbose=0, validation_split=0.2, shuffle=False, callbacks=[early_stop])
            lstm_model.save(model_file_lstm)
        else:
            lstm_model = None
        with open(model_file_pkl, 'wb') as f:
            pickle.dump({'lr': lr_cal, 'xgb': xgb_model, 'mlp': mlp_cal, 'rf': rf_cal, 'svm': svm_cal, 'scaler_lstm': scaler_lstm, 'last_trained': time.time()}, f)
    prob_xgb = xgb_model.predict_proba(np.ascontiguousarray(df[features_xgb].iloc[-1:].fillna(0).values, dtype=np.float64))[0, 1]
    prob_mlp = mlp_cal.predict_proba(np.ascontiguousarray(df[features_mlp].iloc[-1:].fillna(0).values, dtype=np.float64))[0, 1]
    prob_lr = lr_cal.predict_proba(np.ascontiguousarray(df[features_lr].iloc[-1:].fillna(0).values, dtype=np.float64))[0, 1]
    prob_rf = rf_cal.predict_proba(np.ascontiguousarray(df[features_rf].iloc[-1:].fillna(0).values, dtype=np.float64))[0, 1]
    prob_svm = svm_cal.predict_proba(np.ascontiguousarray(df[features_svm].iloc[-1:].fillna(0).values, dtype=np.float64))[0, 1]
    live_lstm_df = df[features_lstm].iloc[-5:].fillna(0)
    if len(live_lstm_df) == 5 and lstm_model is not None:
        X_live_lstm_scaled = scaler_lstm.transform(live_lstm_df.values)
        X_live_seq = np.array([X_live_lstm_scaled])
        prob_lstm = float(lstm_model.predict(X_live_seq, verbose=0)[0][0])
    else:
        prob_lstm = 0.5
    ensemble_prob = (prob_xgb * 0.25) + (prob_rf * 0.25) + (prob_lstm * 0.20) + (prob_mlp * 0.15) + (prob_svm * 0.10) + (prob_lr * 0.05)
    probs = [prob_xgb, prob_rf, prob_lstm, prob_mlp, prob_svm, prob_lr]
    bull_count = sum(1 for p in probs if p >= 0.5)
    bear_count = sum(1 for p in probs if p < 0.5)
    is_bullish = bull_count >= 4
    is_bearish = bear_count >= 4
    base_conf = abs(ensemble_prob - 0.5) * 200
    if is_bullish or is_bearish:
        confidence_score = min(base_conf + 15.0, 99.9)
        agreement_text = "Mayoritas Sepakat"
    else:
        confidence_score = max(base_conf - 10.0, 15.0)
        agreement_text = "Beda Pendapat"
    probs_dict = {"XGBoost": prob_xgb, "RandomForest": prob_rf, "LSTM": prob_lstm, "MLP": prob_mlp, "SVM": prob_svm, "Logistic Reg": prob_lr}
    return ensemble_prob, round(confidence_score, 1), agreement_text, probs_dict

def manage_history_and_evaluate(df, prob_1h, prob_6h, indodax_live_idr, kurs_idr, global_usd):
    waktu_eksekusi = datetime.now(pytz.utc)
    candle_time = df.index[-1]
    target_time_1h = candle_time + pd.Timedelta(hours=1)
    target_time_6h = candle_time + pd.Timedelta(hours=6)
    eval_1h_msg = "Pending"
    eval_6h_msg = "Pending"
    risk_multiplier = 1.0
    file_exists = os.path.isfile(HISTORY_FILE)
    if file_exists:
        try:
            with open(HISTORY_FILE, 'r') as f:
                if 'usd_start_price' not in f.readline():
                    os.remove(HISTORY_FILE)
                    file_exists = False
        except: pass
    try:
        if file_exists:
            history = pd.read_csv(HISTORY_FILE)
            history['target_1h'] = pd.to_datetime(history['target_1h'], format='mixed', utc=True)
            history['target_6h'] = pd.to_datetime(history['target_6h'], format='mixed', utc=True)
            mask_1h = (history['target_1h'] < candle_time) & (history['result_1h'].isna())
            for idx, row in history[mask_1h].iterrows():
                t_target = row['target_1h']
                if t_target in df.index:
                    past_prob = float(row['prob_1h'])
                    past_idr_price = float(row['idr_start_price'])
                    actual_end_price_idr = float(df.loc[t_target, 'Close'])
                    actual_end_price_usd = actual_end_price_idr / kurs_idr
                    arah_asli = "Naik" if actual_end_price_idr > past_idr_price else "Turun" if actual_end_price_idr < past_idr_price else "Stabil"
                    hasil = f"SUCCESS ({arah_asli})" if (past_prob > 0.5 and actual_end_price_idr > past_idr_price) or (past_prob <= 0.5 and actual_end_price_idr <= past_idr_price) else f"FAILED ({arah_asli})"
                    history.loc[idx, 'result_1h'] = hasil
                    history.loc[idx, 'usd_end_price_1h'] = actual_end_price_usd
                    history.loc[idx, 'idr_end_price_1h'] = actual_end_price_idr
                    if t_target == candle_time: eval_1h_msg = hasil
                elif candle_time > t_target + pd.Timedelta(hours=2): history.loc[idx, 'result_1h'] = "DATA MISSING"
            mask_6h = (history['target_6h'] < candle_time) & (history['result_6h'].isna())
            for idx, row in history[mask_6h].iterrows():
                t_target = row['target_6h']
                if t_target in df.index:
                    past_prob = float(row['prob_6h'])
                    past_idr_price = float(row['idr_start_price'])
                    actual_end_price_idr = float(df.loc[t_target, 'Close'])
                    actual_end_price_usd = actual_end_price_idr / kurs_idr
                    arah_asli = "Naik" if actual_end_price_idr > past_idr_price else "Turun" if actual_end_price_idr < past_idr_price else "Stabil"
                    hasil = f"SUCCESS ({arah_asli})" if (past_prob > 0.5 and actual_end_price_idr > past_idr_price) or (past_prob <= 0.5 and actual_end_price_idr <= past_idr_price) else f"FAILED ({arah_asli})"
                    history.loc[idx, 'result_6h'] = hasil
                    history.loc[idx, 'usd_end_price_6h'] = actual_end_price_usd
                    history.loc[idx, 'idr_end_price_6h'] = actual_end_price_idr
                    if t_target == candle_time: eval_6h_msg = hasil
                elif candle_time > t_target + pd.Timedelta(hours=2): history.loc[idx, 'result_6h'] = "DATA MISSING"
            if len(history) > 1000: history = history.tail(1000)
            history.to_csv(HISTORY_FILE, index=False)
            recent_valid = history.dropna(subset=['result_1h']).tail(24)
            recent_valid = recent_valid[~recent_valid['result_1h'].astype(str).str.contains("MISSING")]
            if len(recent_valid) >= 3:
                win_rate = recent_valid['result_1h'].astype(str).str.contains('SUCCESS').sum() / len(recent_valid)
                if win_rate < 0.4: risk_multiplier = 0.1
                elif win_rate < 0.6: risk_multiplier = 0.5
    except Exception as e: diagnostics["csv"] = f"Error: {e}"
    try:
        t_curr_str = waktu_eksekusi.strftime('%Y-%m-%dT%H:%M:%SZ')
        is_duplicate = False
        if file_exists:
            try:
                if len(history) > 0:
                    last_target_dt = pd.to_datetime(history['target_1h'].iloc[-1], utc=True)
                    if last_target_dt.strftime('%Y-%m-%dT%H:%M') == target_time_1h.strftime('%Y-%m-%dT%H:%M'): is_duplicate = True
            except: pass
        if not is_duplicate:
            with open(HISTORY_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists: writer.writerow(['created_at', 'kurs_usd_idr', 'usd_start_price', 'idr_start_price', 'target_1h', 'prob_1h', 'usd_end_price_1h', 'idr_end_price_1h', 'result_1h', 'target_6h', 'prob_6h', 'usd_end_price_6h', 'idr_end_price_6h', 'result_6h'])
                writer.writerow([t_curr_str, kurs_idr, global_usd, indodax_live_idr, target_time_1h.strftime('%Y-%m-%dT%H:%M:%SZ'), prob_1h, '', '', '', target_time_6h.strftime('%Y-%m-%dT%H:%M:%SZ'), prob_6h, '', '', ''])
    except: diagnostics["csv"] = "Failed Writing CSV"
    return eval_1h_msg, eval_6h_msg, risk_multiplier

def generate_web3_dashboard_data(indodax_idr, global_usd, kurs_idr, prob_1h, prob_6h, df, risk_mult, threshold_1h, threshold_6h):
    try:
        if not os.path.isfile(HISTORY_FILE): return
        history = pd.read_csv(HISTORY_FILE)
        hist_1h = history.dropna(subset=['result_1h']).tail(100)
        table_1h = [{"waktu": pd.to_datetime(row['created_at'], format='mixed', utc=True).tz_convert('Asia/Makassar').strftime('%d %b %H:%M'), "prediksi": "UPTREND" if row['prob_1h'] > threshold_1h else "DOWNTREND", "status": str(row['result_1h']).split(' ')[0]} for _, row in hist_1h.iterrows()]
        hist_6h = history.dropna(subset=['result_6h']).tail(100)
        table_6h = [{"waktu": pd.to_datetime(row['created_at'], format='mixed', utc=True).tz_convert('Asia/Makassar').strftime('%d %b %H:%M'), "prediksi": "UPTREND" if row['prob_6h'] > threshold_6h else "DOWNTREND", "status": str(row['result_6h']).split(' ')[0]} for _, row in hist_6h.iterrows()]
        total_1h = len([x for x in table_1h if "SUCCESS" in x['status']])
        web_data = {
            "last_update": datetime.now(pytz.timezone('Asia/Makassar')).strftime('%d %B %Y %H:%M WITA'),
            "prices": {"indodax_idr": indodax_idr, "global_usd": global_usd, "kurs_idr": kurs_idr, "global_idr_converted": global_usd * kurs_idr, "spread_premium": 0},
            "current_prediction": {"prob_1h": prob_1h, "prob_6h": prob_6h, "arah_1h": "UPTREND" if prob_1h > threshold_1h else "DOWNTREND", "arah_6h": "UPTREND" if prob_6h > threshold_6h else "DOWNTREND"},
            "stats": {"win_rate_1h": round((total_1h / len(table_1h) * 100) if table_1h else 0, 1), "risk_multiplier": risk_mult},
            "indicators": {"rsi": round(df['RSI'].iloc[-1], 2), "trend": "UPTREND" if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else "DOWNTREND"},
            "history_1h": table_1h[::-1], "history_6h": table_6h[::-1]
        }
        with open(WEB_DATA_FILE, 'w') as f: json.dump(web_data, f, indent=4)
    except Exception as e: diagnostics["web3"] = f"JSON Export Failed: {e}"

def get_dynamic_thresholds(df, base_1h=0.50, base_6h=0.50):
    vol, vol_mean = df["Volatility"].iloc[-1], df["Volatility"].rolling(100).mean().iloc[-1]
    trend_up = df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
    if vol > vol_mean * 1.5: 
        if trend_up: return base_1h - 0.01, base_6h - 0.01
        else: return base_1h + 0.03, base_6h + 0.04 
    elif vol < vol_mean * 0.5: 
        return base_1h - 0.02, base_6h - 0.03
    return base_1h, base_6h

def calculate_dynamic_buy_size(capital, prob, threshold):
    if capital < MIN_TRADE_IDR: return 0.0
    risk_pct = MAX_RISK_PCT if prob >= threshold + 0.05 else BASE_RISK_PCT
    ideal_size = capital * risk_pct
    if ideal_size < MIN_TRADE_IDR: ideal_size = MIN_TRADE_IDR
    if ideal_size > capital: ideal_size = capital
    return ideal_size

def save_and_copy_chart(fig, base_filename):
    file_path = f"{base_filename}.png"
    fig.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close('all')

def plot_professional_analysis(df, prob_1h, prob_6h, atr, base_filename="chart_main"):
    plt.clf()
    plot_data = df.tail(80).copy()
    plot_data.index = plot_data.index.tz_convert('Asia/Makassar')
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 8))
    plt.plot(plot_data.index, plot_data['Close'], label='Indodax Price (IDR)', color='black', linewidth=2, zorder=5)
    plt.plot(plot_data.index, plot_data['VWAP_24'], label='VWAP', color='#ff7f0e', linestyle='-.')
    try:
        if os.path.exists(HISTORY_FILE):
            hist = pd.read_csv(HISTORY_FILE).tail(15)
            hist['created_at'] = pd.to_datetime(hist['created_at'], utc=True).dt.tz_convert('Asia/Makassar')
            for _, row in hist.iterrows():
                t_past = row['created_at']
                if t_past in plot_data.index:
                    p_start = row['idr_start_price']
                    p_target_1h = p_start + (atr * (float(row['prob_1h']) - 0.5) * 2)
                    plt.scatter(t_past + pd.Timedelta(hours=1), p_target_1h, color='blue', alpha=0.3, s=40)
                    plt.plot([t_past, t_past + pd.Timedelta(hours=1)], [p_start, p_target_1h], color='blue', alpha=0.2, linestyle=':')
    except: pass
    last_time, last_price = plot_data.index[-1], plot_data['Close'].iloc[-1]
    t_1h, p_1h = last_time + pd.Timedelta(hours=1), last_price + (atr * (prob_1h - 0.5) * 2)
    t_6h, p_6h = last_time + pd.Timedelta(hours=6), last_price + (atr * (prob_6h - 0.5) * 4)
    plt.scatter(t_1h, p_1h, color='blue', s=80, marker='o', zorder=10, label='1H Target')
    plt.plot([last_time, t_1h], [last_price, p_1h], color='blue', linestyle='--', linewidth=2)
    plt.scatter(t_6h, p_6h, color='red', s=100, marker='X', zorder=10, label='6H Target')
    plt.plot([last_time, t_6h], [last_price, p_6h], color='red', linestyle='--', linewidth=2)
    plt.title('INDODAX AI STRICT MODE', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', framealpha=0.9)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    save_and_copy_chart(fig, base_filename)

def plot_dashboard_indicators(df, base_filename="chart_indicators"):
    try:
        plt.clf()
        plot_data = df.tail(80).copy()
        plot_data.index = plot_data.index.tz_convert('Asia/Makassar')
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 16))
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(plot_data.index, plot_data['Close'], color='black', label='Harga IDR', linewidth=2)
        ax1.plot(plot_data.index, plot_data['EMA20'], color='blue', alpha=0.8, label='EMA 20')
        ax1.set_title('1. PRICE TREND & EMA (IDR)', fontweight='bold')
        ax1.legend(loc='upper left')
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(plot_data.index, plot_data['RSI'], color='purple', linewidth=2, label='RSI')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.set_title('2. RSI (Momentum)', fontweight='bold')
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.bar(plot_data.index, plot_data['MACD_Hist'], color=np.where(plot_data['MACD_Hist']>0, 'green', 'red'), alpha=0.5)
        ax3.set_title('3. MACD', fontweight='bold')
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(plot_data.index, plot_data['ADX'], color='black', linewidth=2, label='ADX')
        ax4.set_title('4. ADX & DMI', fontweight='bold')
        for ax in [ax1, ax2, ax3, ax4]: ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
        save_and_copy_chart(fig, base_filename)
    except: diagnostics["chart"] = "Render Failed"

def send_telegram_messages(pesan_utama, chart_paths, pesan_diag=""):
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "MASUKKAN_TOKEN_BOT_TELEGRAM_ANDA":
        print(pesan_utama)
        return
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        http_session.post(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': pesan_utama, 'parse_mode': 'Markdown'})
        for path in chart_paths:
            if os.path.exists(path):
                with open(path, 'rb') as photo:
                    http_session.post(url_photo, data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': photo})
    except Exception as e: print(f"Gagal kirim Telegram: {e}")

def get_gemini_insight(prompt):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "MASUKKAN_API_KEY_GEMINI_ANDA": return "Insight AI dinonaktifkan."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.4, "maxOutputTokens": 350}}
    try:
        response = http_session.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e: return "Sistem AI sedang menganalisis pasar."

def main():
    sekarang_wita = datetime.now(pytz.timezone('Asia/Makassar'))
    try:
        data_makro = fetch_global_macro_data()
        fng_dict = fetch_fear_and_greed()
        df, kurs_idr, global_usd, indodax_live_idr = fetch_data_incremental()
        df = engineer_features(df, fng_dict)
        df_closed = df.iloc[:-1].copy()
        current_regime = df["Regime_Str"].iloc[-1]
        buy_wall, sell_wall = fetch_indodax_depth(indodax_live_idr)
        status_berita, daftar_berita = fetch_crypto_news_sentiment()
        current_atr = df["ATR"].iloc[-1]
        
        prob_1h, conf_score_1h, conf_text_1h, details_1h = train_honest_model(df_closed, "Target_1H", 1, "1H", current_regime)
        prob_6h, conf_score_6h, conf_text_6h, details_6h = train_honest_model(df_closed, "Target_6H", 6, "6H", current_regime)
        eval_1h, eval_6h, risk_mult = manage_history_and_evaluate(df, prob_1h, prob_6h, indodax_live_idr, kurs_idr, global_usd)
        threshold_1h, threshold_6h = get_dynamic_thresholds(df)
        generate_web3_dashboard_data(indodax_live_idr, global_usd, kurs_idr, prob_1h, prob_6h, df, risk_mult, threshold_1h, threshold_6h)
        
        fng_val = df['FnG_Index'].iloc[-1]
        rsi_sekarang = df['RSI'].iloc[-1]
        
        dynamic_drop_pct = max(1.5, min(df['Volatility'].iloc[-1] * 250, 6.0))
        dynamic_bounce_pct = max(1.0, dynamic_drop_pct * 0.5)
        
        try:
            with open(ALERT_FILE, 'r') as f: last_state = json.load(f)
            last_price = last_state.get('price', indodax_live_idr)
            last_buy_price = float(last_state.get('last_buy_price', 0.0))
            highest_price_since_buy = float(last_state.get('highest_price_since_buy', 0.0))
            bullets_fired = int(last_state.get('bullets_fired', 0))
            lowest_price_since_drop = float(last_state.get('lowest_price_since_drop', 0.0))
            last_update_id = last_state.get('last_update_id', 0)
        except:
            last_state = {}
            last_price = indodax_live_idr
            last_buy_price = 0.0
            highest_price_since_buy = 0.0
            bullets_fired = 0
            lowest_price_since_drop = 0.0
            last_update_id = 0
            
        aksi_user, harga_input, last_update_id = check_user_telegram_updates(last_update_id)
        if aksi_user == "BUY" and harga_input > 0:
            if last_buy_price == 0:
                last_buy_price = harga_input
                bullets_fired = 1
                highest_price_since_buy = indodax_live_idr
                lowest_price_since_drop = 0.0
            else:
                last_buy_price = (last_buy_price + harga_input) / 2
                bullets_fired += 1
                lowest_price_since_drop = 0.0
        elif aksi_user == "SELL":
            last_buy_price = 0.0
            highest_price_since_buy = 0.0
            bullets_fired = 0
            lowest_price_since_drop = 0.0
            
        saran_tindakan = "WAIT & SEE"
        alasan_saran = "Menunggu momentum yang tepat."
        status_posisi_teks = "Sedang Kosong"
        persentase_perubahan_pnl = 0.0
        
        if last_buy_price == 0:
            kondisi_uptrend = df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]
            volume_valid = df['Volume_Spike'].iloc[-1] > 0.8  
            harga_vs_vwap = (indodax_live_idr - df['VWAP_24'].iloc[-1]) / df['VWAP_24'].iloc[-1]
            tidak_overextended = harga_vs_vwap < 0.03 
            sinyal_uptrend = kondisi_uptrend and prob_1h >= threshold_1h and (40 < rsi_sekarang < 70) and volume_valid and tidak_overextended
            if sinyal_uptrend:
                saran_tindakan, alasan_saran = "BUY PULLBACK", f"Tren Naik, AI Yakin, RSI adem ({rsi_sekarang:.1f})."
            elif prob_1h < 0.40:
                saran_tindakan, alasan_saran = "DOWNTREND", "AI mendeteksi pelemahan."
        else:
            highest_price_since_buy = max(highest_price_since_buy, indodax_live_idr)
            gross_pnl_pct = ((indodax_live_idr - last_buy_price) / last_buy_price) * 100
            persentase_perubahan_pnl = gross_pnl_pct - (FEE_MAKER_TOTAL * 2)
            highest_gross_pct = ((highest_price_since_buy - last_buy_price) / last_buy_price) * 100
            drop_from_peak_pct = ((highest_price_since_buy - indodax_live_idr) / highest_price_since_buy) * 100
            status_posisi_teks = f"Memegang BTC (Peluru: {bullets_fired} | Avg Beli: {format_rupiah(last_buy_price)} | Net PnL: {persentase_perubahan_pnl:+.2f}%)"
            
            kondisi_take_profit = False
            reason_exit = ""
            
            drop_from_avg = ((last_buy_price - indodax_live_idr) / last_buy_price) * 100
            if drop_from_avg >= dynamic_drop_pct:
                if lowest_price_since_drop == 0.0:
                    lowest_price_since_drop = indodax_live_idr
                else:
                    lowest_price_since_drop = min(lowest_price_since_drop, indodax_live_idr)
                    
                bounce_from_bottom = ((indodax_live_idr - lowest_price_since_drop) / lowest_price_since_drop) * 100
                if bounce_from_bottom >= dynamic_bounce_pct:
                    vol_spike = df['Volume_Spike'].iloc[-1]
                    if prob_1h >= threshold_1h and vol_spike > 1.0:
                        saran_tindakan = f"BUY DCA SNIPER (PELURU {bullets_fired + 1})"
                        alasan_saran = f"Harga memantul {bounce_from_bottom:.2f}% dari jurang. Volatilitas butuh drop {dynamic_drop_pct:.1f}%."
                    else:
                        saran_tindakan = "RADAR MODE (STANDBY)"
                        alasan_saran = f"Harga memantul, tapi AI belum yakin atau tidak ada volume bandar."
                else:
                    saran_tindakan = "RADAR MODE (TRACKING DOWN)"
                    alasan_saran = f"Mencari dasar jurang. Syarat pantul: {dynamic_bounce_pct:.1f}%. Jurang: {format_rupiah(lowest_price_since_drop)}"
            
            if "BUY DCA" not in saran_tindakan and "RADAR MODE" not in saran_tindakan:
                if highest_gross_pct >= BREAK_EVEN_TRIGGER_PCT:
                    if persentase_perubahan_pnl <= BREAK_EVEN_LOCK_PCT:
                        kondisi_take_profit = True
                        reason_exit = "BREAK-EVEN LOCK"
                    elif drop_from_peak_pct > TRAILING_PROFIT_DROP_PCT:
                        kondisi_take_profit = True
                        reason_exit = "TRAILING PROFIT"
                        
                kondisi_patah_tren = (df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1]) and (persentase_perubahan_pnl < 0) and not (highest_gross_pct >= BREAK_EVEN_TRIGGER_PCT)
                kondisi_stop_loss = (persentase_perubahan_pnl <= STOP_LOSS_NET)
                
                if kondisi_take_profit:
                    saran_tindakan, alasan_saran = f"TAKE PROFIT ({reason_exit})", f"Smart Lock-In Aktif."
                elif kondisi_patah_tren:
                    saran_tindakan, alasan_saran = "CUT LOSS (TREND PATAH)", "Tren garis EMA patah."
                elif kondisi_stop_loss:
                    saran_tindakan, alasan_saran = "STOP LOSS", f"Batas rugi tercapai ({persentase_perubahan_pnl:.2f}% net)."
                else:
                    saran_tindakan, alasan_saran = "HOLD POSITION", f"Posisi aman. Pucuk: {format_rupiah(highest_price_since_buy)}"
                
        status_orderbook = f"Pembeli Kuat ({(buy_wall/sell_wall if sell_wall>0 else 1):.1f}x)" if buy_wall > sell_wall else f"Penjual Kuat ({(sell_wall/buy_wall if buy_wall>0 else 1):.1f}x)"
        kirim_telegram, alasan_kirim = False, ""
        
        # Pemicu Laporan Startup ke Telegram
        if not os.path.exists(ALERT_FILE):
            kirim_telegram = True
            alasan_kirim = "STARTUP SYSTEM"
            
        raw_diff_pct = ((indodax_live_idr - last_price) / last_price) * 100 if last_price > 0 else 0
        teks_perubahan = f"Naik +{raw_diff_pct:.2f}%" if raw_diff_pct > 0 else f"Turun {raw_diff_pct:.2f}%" if raw_diff_pct < 0 else "Stabil"
        
        if abs(raw_diff_pct) >= 0.6: 
            kirim_telegram, alasan_kirim = True, "Pergerakan Cepat"
        elif saran_tindakan != last_state.get('saran', '') and any(x in saran_tindakan for x in ["BUY", "TAKE PROFIT", "LOSS", "SELL"]): 
            kirim_telegram, alasan_kirim = True, "PERUBAHAN SINYAL AI"
        elif 5 <= sekarang_wita.minute < 20 and last_state.get('last_hourly_report') != sekarang_wita.strftime('%Y-%m-%d-%H'): 
            kirim_telegram, alasan_kirim = True, "Laporan Rutin"
            
        if kirim_telegram or not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "MASUKKAN_TOKEN_BOT_TELEGRAM_ANDA":
            with open(ALERT_FILE, 'w') as f: 
                json.dump({
                    'price': indodax_live_idr, 
                    'saran': saran_tindakan, 
                    'last_buy_price': last_buy_price, 
                    'highest_price_since_buy': highest_price_since_buy if last_buy_price > 0 else 0.0,
                    'bullets_fired': bullets_fired,
                    'lowest_price_since_drop': lowest_price_since_drop if last_buy_price > 0 else 0.0,
                    'last_hourly_report': sekarang_wita.strftime('%Y-%m-%d-%H'),
                    'last_update_id': last_update_id
                }, f)
            plot_professional_analysis(df, prob_1h, prob_6h, current_atr, "chart_main")
            plot_dashboard_indicators(df, "chart_indicators")
            icon_aksi = "🟢" if "BUY" in saran_tindakan else "🔴" if any(x in saran_tindakan for x in ["SELL", "PROFIT", "LOSS"]) else "🟡"
            
            # KODE BARU (DEEP ANALYSIS)
            macd_status = "Positif (Bullish Momentum)" if df['MACD_Hist'].iloc[-1] > 0 else "Negatif (Bearish Momentum)"
            teks_orderbook = f"Dominasi Pembeli ({(buy_wall/sell_wall if sell_wall>0 else 1):.1f}x)" if buy_wall > sell_wall else f"Dominasi Penjual ({(sell_wall/buy_wall if buy_wall>0 else 1):.1f}x)"
            teks_headlines = " | ".join(daftar_berita[:3]) if daftar_berita else "Tidak ada berita signifikan"
            
            if last_buy_price > 0 and persentase_perubahan_pnl >= 0.1:
                prompt_final = f"""
                Anda adalah Senior Kuantitatif Analis di firma investasi kripto. Analisis posisi trading ini dan berikan saran strategis (maksimal 3 kalimat):
                - Status Posisi: Sedang HOLD BTC. Net Profit: {persentase_perubahan_pnl:+.2f}%
                - Kondisi Harga: {format_rupiah(indodax_live_idr)} (Rezim: {current_regime})
                - Indikator Teknikal: RSI {rsi_sekarang:.1f}, MACD {macd_status}
                - Psikologi & Bandar: Fear & Greed {int(fng_val)}/100, Orderbook {teks_orderbook}, Berita {status_berita}
                - Makro Ekonomi: Dolar DXY {data_makro['DXY']}, Saham S&P500 {data_makro['SP500']}
                - HEADLINES BERITA SAAT INI: {teks_headlines}
                Pertanyaan: Berdasarkan probabilitas pembalikan arah dari data di atas, haruskah saya Take Profit sekarang atau terus Hold (Trailing)?
                """
            else:
                prompt_final = f"""
                Anda adalah Senior Kuantitatif Analis di firma investasi kripto. Berikan kesimpulan pasar (maksimal 3 kalimat) untuk strategi Scalping/DCA:
                - Kondisi Harga BTC: {format_rupiah(indodax_live_idr)} ({teks_perubahan}). Rezim pasar saat ini: {current_regime}.
                - Indikator Teknikal: RSI {rsi_sekarang:.1f}, MACD {macd_status}
                - Psikologi & Bandar: Fear & Greed {int(fng_val)}/100, Orderbook {teks_orderbook}, Berita {status_berita}
                - Makro Ekonomi: Dolar AS (DXY) {data_makro['DXY']}, Saham AS (S&P500) {data_makro['SP500']}
                - HEADLINES BERITA SAAT INI: {teks_headlines}
                Pertanyaan: Apa narasi utama yang sedang menggerakkan pasar detik ini, dan apa posisi terbaik (Beli/Tunggu) secara fundamental?
                """
            
            insight_gemini = get_gemini_insight(prompt_final) if kirim_telegram else "Menunggu AI..."
            
            pesan_utama = f"🚨 *LAPORAN PASAR AI* 🚨\n⏰ {sekarang_wita.strftime('%d %b %Y | %H:%M')}\n🔄 Trigger: *{alasan_kirim}*\n\n"
            pesan_utama += f"🧠 *GEMMA INSIGHT:*\n_{insight_gemini}_\n\n"
            pesan_utama += f"📦 *STATUS:*\n{status_posisi_teks}\n\n"
            pesan_utama += f"📊 *AKSI:* {icon_aksi} *{saran_tindakan}*\n💡 {alasan_saran}\n\n"
            if "BUY" in saran_tindakan:
                estimasi_jual = indodax_live_idr * 1.02 
                pesan_utama += f"🕸️ *REKOMENDASI BELI:* Gunakan {BASE_RISK_PCT*100}% Sisa Saldo (Min. Rp25rb)\n"
                pesan_utama += f"🎯 *ESTIMASI JUAL:* {format_rupiah(estimasi_jual)}\n\n"
            pesan_utama += f"💰 *HARGA:* {format_rupiah(indodax_live_idr)}\n"
            pesan_utama += f"🌊 *PASAR:* {current_regime} | F&G: {int(fng_val)}\n\n"
            pesan_utama += f"🌍 *MAKRO:* DXY: {data_makro['DXY']} | S&P500: {data_makro['SP500']}\n\n"
            pesan_utama += f"🤖 *PREDIKSI LOKAL:* 1H: *{prob_1h*100:.1f}%* | 6H: *{prob_6h*100:.1f}%*\n\n"
            pesan_utama += f"⚖️ *ORDERBOOK:* {status_orderbook}"
            send_telegram_messages(pesan_utama, ["chart_main.png", "chart_indicators.png"])
    except Exception as fatal_e:
        print(f"FATAL ERROR: {fatal_e}")

def run_offline_backtest(initial_capital=BACKTEST_INITIAL_CAPITAL, test_size=BACKTEST_TEST_SIZE):
    print("🚀 Memulai Backtest Engine ULTIMATE SMART SIZING DCA...")
    if not os.path.exists(OHLCV_FILE):
        print(f"❌ File tidak ditemukan. Jalankan mode Live dulu untuk narik data.")
        return
    df = pd.read_csv(OHLCV_FILE, index_col='Time', parse_dates=True)
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    print(f"📦 Total Data Dimuat: {len(df)} candles.")
    dummy_fng = {df.index[i].date(): 50.0 for i in range(len(df))}
    df = engineer_features(df, dummy_fng)
    df.dropna(inplace=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print("🧠 Melatih 6 Model AI Ensemble...")
    features_xgb = ["EMA_Spread", "MACD_Hist", "Trend_Slope", "Momentum_Accel", "ADX"]
    features_mlp = ["Volatility", "ATR_Ratio", "BB_Width", "Log_Return"]
    features_lr = ["EMA_Spread", "BB_Width", "Trend_Slope", "Log_Return", "RSI"]
    features_rf = ["Volume_Spike", "BB_Width", "ATR_Ratio", "Log_Return", "RSI"]
    features_svm = ["Regime", "EMA_Spread", "Volatility", "Trend_Slope", "MACD_Hist"]
    features_lstm = ["Log_Return", "RSI", "MACD_Hist", "Volatility", "Volume_Spike"]
    y_train = train_df["Target_1H"].astype(int).values
    tscv = TimeSeriesSplit(n_splits=3) 
    xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, subsample=0.6, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(train_df[features_xgb].values, y_train)
    rf_pipeline = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42, n_jobs=-1))])
    rf_cal = CalibratedClassifierCV(rf_pipeline, method="sigmoid", cv=tscv)
    rf_cal.fit(train_df[features_rf].values, y_train)
    lr_pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=300, C=0.5, random_state=42))])
    lr_cal = CalibratedClassifierCV(lr_pipeline, method="sigmoid", cv=tscv)
    lr_cal.fit(train_df[features_lr].values, y_train)
    mlp_pipeline = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(16, 8), alpha=0.01, max_iter=300, random_state=42))])
    mlp_cal = CalibratedClassifierCV(mlp_pipeline, method="sigmoid", cv=tscv)
    mlp_cal.fit(train_df[features_mlp].values, y_train)
    svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, C=0.5, random_state=42))])
    svm_cal = CalibratedClassifierCV(svm_pipeline, method="sigmoid", cv=tscv)
    svm_cal.fit(train_df[features_svm].values, y_train)
    scaler_lstm = StandardScaler()
    X_train_lstm = scaler_lstm.fit_transform(train_df[features_lstm].values)
    X_seq_train, y_seq_train = build_lstm_data(X_train_lstm, y_train, 5)
    lstm_model = Sequential()
    lstm_model.add(Input(shape=(5, len(features_lstm))))
    lstm_model.add(LSTM(8, activation='relu'))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
    lstm_model.fit(X_seq_train, y_seq_train, epochs=5, batch_size=32, verbose=0)
    print("⚙️ Menerapkan Prediksi...")
    prob_xgb = xgb_model.predict_proba(test_df[features_xgb].values)[:, 1]
    prob_rf = rf_cal.predict_proba(test_df[features_rf].values)[:, 1]
    prob_lr = lr_cal.predict_proba(test_df[features_lr].values)[:, 1]
    prob_mlp = mlp_cal.predict_proba(test_df[features_mlp].values)[:, 1]
    prob_svm = svm_cal.predict_proba(test_df[features_svm].values)[:, 1]
    X_test_lstm = scaler_lstm.transform(test_df[features_lstm].values)
    padded_X_test = np.vstack([X_train_lstm[-5:], X_test_lstm])
    X_seq_test, _ = build_lstm_data(padded_X_test, np.zeros(len(padded_X_test)), 5)
    prob_lstm = lstm_model.predict(X_seq_test, verbose=0).flatten()
    test_df['Prob_1H'] = (prob_xgb * 0.25) + (prob_rf * 0.25) + (prob_lstm * 0.20) + (prob_mlp * 0.15) + (prob_svm * 0.10) + (prob_lr * 0.05)
    
    capital = initial_capital
    position = 0.0
    avg_buy_price = 0.0
    highest_price_since_buy = 0.0 
    bullets_fired = 0
    lowest_price_since_drop = 0.0
    total_invested = 0.0
    trade_history = []
    prob_dynamic_threshold = test_df['Prob_1H'].quantile(0.75) 
    print(f"\n📈 Memulai Simulasi Trading Mode DYNAMIC DCA... Modal Awal: {format_rupiah(capital)}")
    
    for i in range(1, len(test_df)):
        current_candle = test_df.iloc[i]
        price = current_candle['Close']
        prob = current_candle['Prob_1H']
        rsi = current_candle['RSI']
        kondisi_uptrend = current_candle['EMA20'] > current_candle['EMA50']
        dynamic_drop_pct = max(1.5, min(current_candle['Volatility'] * 250, 6.0))
        dynamic_bounce_pct = max(1.0, dynamic_drop_pct * 0.5)
        
        if position == 0:
            sinyal_uptrend = kondisi_uptrend and prob >= prob_dynamic_threshold and (40 < rsi < 70)
            if sinyal_uptrend:
                buy_cost = calculate_dynamic_buy_size(capital, prob, prob_dynamic_threshold)
                if buy_cost >= MIN_TRADE_IDR:
                    btc_bought = buy_cost / price
                    fee_rp = buy_cost * (FEE_MAKER_TOTAL / 100)
                    position = btc_bought
                    capital -= (buy_cost + fee_rp)
                    avg_buy_price = price
                    highest_price_since_buy = price 
                    bullets_fired = 1
                    lowest_price_since_drop = 0.0
                    total_invested = buy_cost
                    trade_history.append({'time': test_df.index[i], 'type': f'BUY (Peluru 1)', 'price': price, 'capital': capital, 'port_val': capital + (position * price)})
                    
        elif position > 0:
            if capital >= MIN_TRADE_IDR:
                drop_from_avg = ((avg_buy_price - price) / avg_buy_price) * 100
                if drop_from_avg >= dynamic_drop_pct:
                    if lowest_price_since_drop == 0.0: lowest_price_since_drop = price
                    else: lowest_price_since_drop = min(lowest_price_since_drop, price)
                    bounce_from_bottom = ((price - lowest_price_since_drop) / lowest_price_since_drop) * 100
                    if bounce_from_bottom >= dynamic_bounce_pct:
                        vol_spike = current_candle['Volume_Spike']
                        if prob >= prob_dynamic_threshold and vol_spike > 1.0:
                            buy_cost = calculate_dynamic_buy_size(capital, prob, prob_dynamic_threshold)
                            if buy_cost >= MIN_TRADE_IDR:
                                btc_bought = buy_cost / price
                                fee_rp = buy_cost * (FEE_MAKER_TOTAL / 100)
                                total_invested += buy_cost
                                position += btc_bought
                                capital -= (buy_cost + fee_rp)
                                avg_buy_price = total_invested / position
                                bullets_fired += 1
                                highest_price_since_buy = price 
                                lowest_price_since_drop = 0.0
                                trade_history.append({'time': test_df.index[i], 'type': f'BUY DCA (Peluru {bullets_fired})', 'price': price, 'capital': capital, 'port_val': capital + (position * price)})
                                continue 
            
            highest_price_since_buy = max(highest_price_since_buy, price)
            gross_pnl_pct = ((price - avg_buy_price) / avg_buy_price) * 100
            total_fee_pct = FEE_MAKER_TOTAL * 2 
            net_pnl_pct = gross_pnl_pct - total_fee_pct
            highest_gross_pct = ((highest_price_since_buy - avg_buy_price) / avg_buy_price) * 100
            drop_from_peak_pct = ((highest_price_since_buy - price) / highest_price_since_buy) * 100
            kondisi_take_profit = False
            reason = ""
            if highest_gross_pct >= BREAK_EVEN_TRIGGER_PCT:
                if net_pnl_pct <= BREAK_EVEN_LOCK_PCT:
                    kondisi_take_profit = True
                    reason = "BREAK-EVEN LOCK"
                elif drop_from_peak_pct > TRAILING_PROFIT_DROP_PCT:
                    kondisi_take_profit = True
                    reason = "TRAILING PROFIT"
            kondisi_patah_tren = (current_candle['EMA20'] < current_candle['EMA50']) and (net_pnl_pct < 0) and not (highest_gross_pct >= BREAK_EVEN_TRIGGER_PCT)
            kondisi_stop_loss = (net_pnl_pct <= STOP_LOSS_NET)
            if kondisi_take_profit or kondisi_patah_tren or kondisi_stop_loss:
                gross_sell = position * price
                fee_rp = gross_sell * (FEE_MAKER_TOTAL / 100)
                net_sell = gross_sell - fee_rp
                capital += net_sell
                position = 0.0
                highest_price_since_buy = 0.0 
                avg_buy_price = 0.0
                bullets_fired = 0
                lowest_price_since_drop = 0.0
                total_invested = 0.0
                if not reason: reason = "TREND PATAH" if kondisi_patah_tren else "STOP LOSS"
                trade_history.append({'time': test_df.index[i], 'type': f'SELL ({reason})', 'price': price, 'pnl': net_pnl_pct, 'capital': capital, 'port_val': capital})
                
    if position > 0:
        final_price = test_df.iloc[-1]['Close']
        capital += (position * final_price) * (1 - (FEE_MAKER_TOTAL / 100))
        trade_history.append({'time': test_df.index[-1], 'type': 'SELL (END)', 'price': final_price, 'capital': capital, 'port_val': capital})
    calculate_and_plot_metrics(initial_capital, capital, trade_history, test_df)

def calculate_and_plot_metrics(initial_capital, final_capital, trade_history, test_df):
    sells = [t for t in trade_history if 'SELL' in t['type'] and 'pnl' in t]
    total_trades = len(sells)
    if total_trades > 0:
        wins = len([t for t in sells if t['pnl'] > 0])
        win_rate = (wins / total_trades) * 100
    else: win_rate = 0.0
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    print("\n" + "="*40 + "\n📊 HASIL BACKTEST KESELURUHAN\n" + "="*40)
    print(f"Periode Test   : {test_df.index[0].date()} s/d {test_df.index[-1].date()}")
    print(f"Modal Awal     : {format_rupiah(initial_capital)}")
    print(f"Modal Akhir    : {format_rupiah(final_capital)}")
    print(f"Total Return   : {total_return:+.2f}%")
    print(f"Total Trading  : {total_trades} kali (Siklus Jual)\nWin Rate       : {win_rate:.2f}%\n" + "="*40)
    try:
        df_trades = pd.DataFrame(trade_history)
        if not df_trades.empty:
            plt.figure(figsize=(15, 8))
            plt.style.use('seaborn-v0_8-darkgrid')
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(test_df.index, test_df['Close'], color='gray', alpha=0.3, label='Harga BTC')
            ax1.plot(df_trades['time'], df_trades['port_val'], color='blue', linewidth=2, alpha=0.6, label='Portfolio Value')
            buys = df_trades[df_trades['type'].str.contains('BUY')]
            sells = df_trades[df_trades['type'].str.contains('SELL')]
            if not buys.empty:
                ax2.scatter(buys['time'], buys['price'], color='green', marker='^', s=120, label='ENTRY BELI', zorder=5)
            if not sells.empty:
                ax2.scatter(sells['time'], sells['price'], color='red', marker='v', s=120, label='EXIT JUAL', zorder=5)
            ax1.axhline(initial_capital, color='black', linestyle='--', alpha=0.5, label='Garis Modal Awal')
            ax1.set_title('EQUITY CURVE & ENTRY/EXIT POINTS (SNIPER MODE)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Sisa Saldo Tunai (IDR)')
            ax2.set_ylabel('Harga Eksekusi BTC (IDR)')
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', framealpha=0.9)
            plt.savefig("backtest_result.png", dpi=150, bbox_inches='tight')
            print("📸 Grafik Equity Curve berhasil disimpan")
    except Exception as e: print(f"Gagal memplot grafik: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--backtest':
        run_offline_backtest(initial_capital=BACKTEST_INITIAL_CAPITAL, test_size=BACKTEST_TEST_SIZE) 
    else:
        print("🚀 Memulai AI Sniper Bot secara LIVE di GitHub Actions...")
        # CATATAN PENTING: Untuk GitHub Actions, kita menghapus loop `while True`.
        # Biarkan GitHub YAML Cron Job yang mengulangnya setiap 5 menit agar file tersimpan.
        main()
        print("✅ Eksekusi selesai. Menunggu cron job GitHub Actions berjalan lagi 5 menit ke depan...")
