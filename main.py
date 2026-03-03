# ==============================================================================
# SISTEM DAY TRADING BITCOIN (TIMEFRAME 1 JAM) - VERSI INDODAX (IDR)
# TERINTEGRASI DENGAN TELEGRAM BOT - ZONA WAKTU WITA
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Wajib untuk server tanpa layar/GUI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import requests
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    os.system('pip install yfinance')
    import yfinance as yf

# ================= KONFIGURASI TELEGRAM =================
# Isi dengan Token Bot dan Chat ID Anda
TELEGRAM_BOT_TOKEN = '8281574109:AAHMCWUiDCGID6zropl3TT0mW5yUtUZK1Gs'
TELEGRAM_CHAT_ID = '8067218202'
# ========================================================

def format_rupiah(angka):
    return f"Rp {angka:,.0f}".replace(',', '.')

def fetch_intraday_data(period='60d', interval='1h'):
    df = yf.download('BTC-USD', period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Konversi Zona Waktu ke WITA (Waktu Indonesia Tengah / Asia/Makassar)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Asia/Makassar') # <-- DIUBAH KE WITA

    try:
        idr_data = yf.download('IDR=X', period='5d', progress=False)
        if isinstance(idr_data.columns, pd.MultiIndex):
            idr_data.columns = idr_data.columns.droplevel(1)
        kurs_idr = float(idr_data['Close'].iloc[-1])
    except Exception as e:
        kurs_idr = 16000.0

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col] * kurs_idr

    return df

def calculate_indicators(df):
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']

    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_20 * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_20 * 2)

    return df

def generate_signals(df):
    scores = []
    for i in range(len(df)):
        score = 0
        if pd.notna(df['MA_50'].iloc[i]) and pd.notna(df['MA_200'].iloc[i]):
            if df['MA_50'].iloc[i] > df['MA_200'].iloc[i]: score += 1
            else: score -= 1

        if pd.notna(df['RSI'].iloc[i]):
            if df['RSI'].iloc[i] < 30: score += 2
            elif df['RSI'].iloc[i] > 70: score -= 2

        if pd.notna(df['MACD_Line'].iloc[i]) and pd.notna(df['Signal_Line'].iloc[i]):
            if df['MACD_Line'].iloc[i] > df['Signal_Line'].iloc[i]: score += 1
            else: score -= 1

        if pd.notna(df['BB_Lower'].iloc[i]) and pd.notna(df['BB_Upper'].iloc[i]):
            if df['Close'].iloc[i] <= df['BB_Lower'].iloc[i]: score += 1
            elif df['Close'].iloc[i] >= df['BB_Upper'].iloc[i]: score -= 1

        scores.append(score)

    df['Signal_Score'] = scores

    conditions = [
        (df['Signal_Score'] >= 3),
        (df['Signal_Score'] > 0) & (df['Signal_Score'] < 3),
        (df['Signal_Score'] == 0),
        (df['Signal_Score'] < 0) & (df['Signal_Score'] > -3),
        (df['Signal_Score'] <= -3)
    ]
    choices = ['STRONG BUY (SANGAT BAGUS)', 'BUY (BOLEH BELI)', 'HOLD (PANTAU)', 'SELL (JUAL)', 'STRONG SELL (JUAL SEMUA)']
    df['Recommendation'] = np.select(conditions, choices, default='HOLD (PANTAU)')

    df = df.dropna(subset=['MA_200'])
    return df

def plot_analysis(df, filename="chart.png"):
    plot_data = df.tail(120)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1.plot(plot_data.index, plot_data['Close'], label='Harga BTC (IDR)', color='black', linewidth=1.5)
    ax1.plot(plot_data.index, plot_data['MA_50'], label='MA 50 Jam', color='blue', linestyle='--')
    ax1.plot(plot_data.index, plot_data['MA_200'], label='MA 200 Jam', color='red', linestyle='--')
    ax1.fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], color='gray', alpha=0.1, label='Bollinger Bands')

    buy_signals = plot_data[plot_data['Recommendation'].str.contains('STRONG BUY')]
    sell_signals = plot_data[plot_data['Recommendation'].str.contains('STRONG SELL')]

    ax1.scatter(buy_signals.index, buy_signals['Close'] * 0.99, marker='^', color='green', s=150, label='STRONG BUY', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['Close'] * 1.01, marker='v', color='red', s=150, label='STRONG SELL', zorder=5)

    ax1.set_title('Analisis DAY TRADING Bitcoin (Timeframe 1 Jam) - WITA', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Harga (Rupiah)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(',', '.')))

    myFmt = mdates.DateFormatter('%d %b\n%H:%M')
    ax1.xaxis.set_major_formatter(myFmt)

    ax2.plot(plot_data.index, plot_data['RSI'], color='purple', label='RSI (14 Jam)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.fill_between(plot_data.index, plot_data['RSI'], 70, where=(plot_data['RSI'] >= 70), color='red', alpha=0.3, interpolate=True)
    ax2.fill_between(plot_data.index, plot_data['RSI'], 30, where=(plot_data['RSI'] <= 30), color='green', alpha=0.3, interpolate=True)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(myFmt)

    ax3.plot(plot_data.index, plot_data['MACD_Line'], color='blue', label='MACD Line')
    ax3.plot(plot_data.index, plot_data['Signal_Line'], color='orange', label='Signal Line')
    colors = ['green' if val >= 0 else 'red' for val in plot_data['MACD_Histogram']]
    ax3.bar(plot_data.index, plot_data['MACD_Histogram'], color=colors, alpha=0.5)
    ax3.set_ylabel('MACD')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(myFmt)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight') # Simpan sebagai gambar, jangan di-show()
    plt.close()

def send_to_telegram(message, image_path):
    print("Mengirim data ke Telegram...")
    # Kirim Pesan Teks
    url_message = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload_msg = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    requests.post(url_message, data=payload_msg)

    # Kirim Gambar Grafik
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        payload_photo = {'chat_id': TELEGRAM_CHAT_ID}
        requests.post(url_photo, data=payload_photo, files={'photo': photo})
    print("Berhasil terkirim!")

def main():
    if TELEGRAM_BOT_TOKEN == 'MASUKKAN_TOKEN_BOT_ANDA_DI_SINI':
        print("PENTING: Anda belum memasukkan Token Bot Telegram di dalam kode!")
        return

    df = fetch_intraday_data(period='60d', interval='1h')
    df = calculate_indicators(df)
    df = generate_signals(df)

    if len(df) > 0:
        latest = df.iloc[-1]
        
        # Susun pesan untuk Telegram
        pesan = f"📊 *STATUS PASAR BITCOIN (1 JAM TERAKHIR)* 📊\n\n"
        pesan += f"🕒 *Waktu Analisis:* {latest.name.strftime('%d-%m-%Y %H:%M WITA')}\n"
        pesan += f"💰 *Harga BTC:* {format_rupiah(latest['Close'])}\n\n"
        
        rsi_status = "(Overbought 🔴)" if latest['RSI'] > 70 else "(Oversold 🟢)" if latest['RSI'] < 30 else "(Netral ⚪)"
        pesan += f"📈 *RSI (14 Jam):* {latest['RSI']:.2f} {rsi_status}\n"
        
        macd_status = "Bullish 🟢" if latest['MACD_Line'] > latest['Signal_Line'] else "Bearish 🔴"
        pesan += f"📊 *MACD:* {macd_status}\n"
        
        trend = "Naik 🟢" if latest['Close'] > latest['MA_50'] else "Turun 🔴"
        pesan += f"📉 *Tren (MA50):* {trend}\n\n"
        
        pesan += f"🚀 *KESIMPULAN: {latest['Recommendation']}*\n\n"
        pesan += "_Bot by GitHub Actions_"

        # Buat grafik dan simpan dengan nama chart.png
        chart_filename = "chart.png"
        plot_analysis(df, chart_filename)

        # Kirim ke Telegram
        send_to_telegram(pesan, chart_filename)
    else:
        print("Data tidak cukup.")

if __name__ == "__main__":
    main()