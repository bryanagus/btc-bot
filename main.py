# ==============================================================================
# BTC QUANT GODMODE ENGINE v6.0
# ABSOLUTE MAX LEVEL - INSTITUTIONAL DESK STYLE
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import pytz
import requests
import time
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

TELEGRAM_BOT_TOKEN = "YOUR_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# ==============================================================================
# DATA
# ==============================================================================
def fetch_data():
    df = yf.download("BTC-USD", period="180d", interval="1h", progress=False)

    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Makassar")

    idr = yf.download("IDR=X", period="5d", progress=False)
    kurs = float(idr["Close"].iloc[-1])

    for col in ["Open","High","Low","Close"]:
        df[col] *= kurs

    return df

# ==============================================================================
# FEATURE ENGINEERING MAX
# ==============================================================================
def create_features(df):

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA_Spread"] = (df["EMA20"] - df["EMA50"]) / df["Close"]

    df["RSI"] = 100 - (100/(1 + (df["Close"].diff().clip(lower=0).rolling(14).mean() /
                                 (-df["Close"].diff().clip(upper=0).rolling(14).mean()))))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    df["MACD_Hist"] = macd - macd.ewm(span=9).mean()

    df["Return_1H"] = df["Close"].pct_change()
    df["Return_3H"] = df["Close"].pct_change(3)
    df["Return_6H"] = df["Close"].pct_change(6)

    df["Volatility"] = (df["High"] - df["Low"]).rolling(14).mean() / df["Close"]
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(24).mean()

    df["Trend_Slope"] = df["Close"].rolling(12).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )

    df["Momentum_Accel"] = df["Return_1H"].diff()

    df["Regime"] = np.where(df["EMA20"] > df["EMA50"], 1, 0)

    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    df = df.dropna()

    features = [
        "EMA_Spread","RSI","MACD_Hist",
        "Return_1H","Return_3H","Return_6H",
        "Volatility","Volume_Ratio",
        "Trend_Slope","Momentum_Accel","Regime"
    ]

    return df, features

# ==============================================================================
# ENSEMBLE + CALIBRATION
# ==============================================================================
def train_models(df, features):

    X = df[features]
    y = df["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    lr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=400)
    gb = GradientBoostingClassifier()

    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, y_train = X_scaled[train_idx], y.iloc[train_idx]
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)

    lr = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
    lr.fit(X_scaled, y)

    prob_lr = lr.predict_proba(X_scaled)[:,1]
    prob_rf = rf.predict_proba(X_scaled)[:,1]
    prob_gb = gb.predict_proba(X_scaled)[:,1]

    ensemble_prob = (prob_lr + prob_rf + prob_gb) / 3

    accuracy = accuracy_score(y, (ensemble_prob>0.5).astype(int)) * 100

    return scaler, ensemble_prob, accuracy

# ==============================================================================
# MONTE CARLO + VAR
# ==============================================================================
def monte_carlo(price, vol, steps=24, sims=2000):

    paths = []
    for _ in range(sims):
        prices = [price]
        for _ in range(steps):
            shock = np.random.normal(0, vol)
            prices.append(prices[-1]*(1+shock))
        paths.append(prices)

    paths = np.array(paths)
    final_prices = paths[:,-1]

    expected = np.mean(final_prices)
    var95 = np.percentile(final_prices,5)

    return expected, var95

# ==============================================================================
# BACKTEST ADVANCED
# ==============================================================================
def backtest(df, probs):

    capital = 100
    equity = [capital]
    wins = 0
    trades = 0
    gross_profit = 0
    gross_loss = 0

    for i in range(len(probs)-1):
        if probs[i] > 0.55:
            ret = df["Return_1H"].iloc[i+1]
            capital *= (1+ret)
            trades+=1
            if ret>0:
                wins+=1
                gross_profit+=ret
            else:
                gross_loss+=abs(ret)
        equity.append(capital)

    equity = np.array(equity)
    returns = pd.Series(equity).pct_change().dropna()

    sharpe = (returns.mean()/returns.std())*np.sqrt(24*365) if returns.std()!=0 else 0
    downside = returns[returns<0]
    sortino = (returns.mean()/downside.std())*np.sqrt(24*365) if downside.std()!=0 else 0

    peak = np.maximum.accumulate(equity)
    drawdown = (equity-peak)/peak
    max_dd = drawdown.min()*100

    cagr = ((equity[-1]/100)**(365/180)-1)*100
    winrate = (wins/trades)*100 if trades>0 else 0
    profit_factor = gross_profit/gross_loss if gross_loss!=0 else 0

    return equity[-1], sharpe, sortino, max_dd, cagr, winrate, profit_factor

# ==============================================================================
# RISK ENGINE
# ==============================================================================
def position_sizing(prob, volatility):

    edge = prob - (1-prob)
    kelly = max(edge,0)

    vol_adjust = min(1/(volatility*100),1)

    size = kelly * vol_adjust
    size = min(size,0.25)  # max 25% exposure

    return round(size*100,2)

# ==============================================================================
# TELEGRAM
# ==============================================================================
def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"Markdown"}
    requests.post(url, data=payload)

# ==============================================================================
# MAIN
# ==============================================================================
def main():

    start = time.time()

    df = fetch_data()
    df, features = create_features(df)

    scaler, probs, accuracy = train_models(df, features)

    latest_prob = probs[-1]
    direction = "NAIK 📈" if latest_prob>0.5 else "TURUN 📉"
    confidence = max(latest_prob,1-latest_prob)*100

    expected, var95 = monte_carlo(
        df["Close"].iloc[-1],
        df["Volatility"].iloc[-1]
    )

    final_capital, sharpe, sortino, max_dd, cagr, winrate, pf = backtest(df, probs)

    exposure = position_sizing(latest_prob, df["Volatility"].iloc[-1])

    runtime = round(time.time()-start,2)

    message = f"""
🏦 *BTC QUANT GODMODE ENGINE v6*

🕒 {df.index[-1].strftime('%d %B %Y | %H:%M WITA')}

💰 Harga Sekarang : Rp {df['Close'].iloc[-1]:,.0f}

🤖 Ensemble Probability : {latest_prob*100:.2f}%
🎯 Direction            : *{direction}*
🔥 AI Confidence        : *{confidence:.2f}%*

🔮 Monte Carlo 24H Mean : Rp {expected:,.0f}
⚠️ VaR 95%              : Rp {var95:,.0f}

📊 Strategy Metrics:
• Accuracy  : {accuracy:.2f}%
• Winrate   : {winrate:.2f}%
• CAGR      : {cagr:.2f}%
• Sharpe    : {sharpe:.2f}
• Sortino   : {sortino:.2f}
• Max DD    : {max_dd:.2f}%
• ProfitFactor : {pf:.2f}

💰 Suggested Exposure : {exposure}% modal

⏱ Runtime : {runtime} detik

_System: Ensemble ML + Calibration + Monte Carlo + VaR + Kelly + Risk Analytics_
"""

    send_to_telegram(message)

if __name__ == "__main__":
    main()
