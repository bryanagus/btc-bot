# ==============================================================================
# SISTEM ALGORITMA TRADING BITCOIN PRO MAX (TIMEFRAME 1 JAM)
# Fitur: Multi-Source News Aggregator, ADX, VWAP, StochRSI, Machine Learning
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

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    os.system('pip install yfinance')
    import yfinance as yf

# ================= KONFIGURASI TELEGRAM =================
TELEGRAM_BOT_TOKEN = '8281574109:AAHMCWUiDCGID6zropl3TT0mW5yUtUZK1Gs'
TELEGRAM_CHAT_ID = '8067218202'
# ========================================================

def format_rupiah(angka):
    if pd.isna(angka): return "Rp 0"
    return f"Rp {angka:,.0f}".replace(',', '.')

def fetch_crypto_news_sentiment():
    """Mengambil berita dari BANYAK sumber dan memfilter duplikat."""
    print("[*] Mengumpulkan dan menganalisis berita Kripto global (Anti-Duplikat)...")
    
    rss_urls = [
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://cointelegraph.com/rss',
        'https://cryptopotato.com/feed/'
    ]
    
    bullish_keywords = ['surge', 'jump', 'rise', 'bull', 'high', 'adopt', 'approve', 'gain', 'positive', 'buy', 'up', 'soar', 'breakout', 'record']
    bearish_keywords = ['drop', 'fall', 'crash', 'bear', 'low', 'ban', 'reject', 'lose', 'negative', 'sell', 'down', 'hack', 'scam', 'plunge']
    
    bullish_score = 0
    bearish_score = 0
    unique_news = set() # Menggunakan Set untuk menghindari duplikat
    total_berita = 0
    
    for url in rss_urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=5)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            # Ambil 10 berita dari tiap sumber
            for item in root.findall('.//item')[:10]:
                title = item.find('title').text
                if not title: continue
                
                # Normalisasi teks (hapus tanda baca, jadikan huruf kecil)
                clean_title = re.sub(r'[^\w\s]', '', title.lower())
                
                # Cek apakah berita mirip/sama sudah pernah dianalisis
                if clean_title not in unique_news:
                    unique_news.add(clean_title)
                    total_berita += 1
                    
                    for word in bullish_keywords:
                        if word in clean_title: bullish_score += 1
                    for word in bearish_keywords:
                        if word in clean_title: bearish_score += 1
        except Exception as e:
            continue # Abaikan sumber yang error, lanjut ke sumber berikutnya

    # Logika Penilaian Sentimen yang Diperbaiki
    selisih = bullish_score - bearish_score
    if selisih >= 3:
        status = f"SANGAT POSITIF 🚀 ({total_berita} Berita Unik)"
        poin = 2
    elif selisih > 0:
        status = f"POSITIF RINGAN 🟢 ({total_berita} Berita Unik)"
        poin = 1
    elif selisih <= -3:
        status = f"SANGAT NEGATIF 🚨 ({total_berita} Berita Unik)"
        poin = -2
    elif selisih < 0:
        status = f"NEGATIF RINGAN 🔴 ({total_berita} Berita Unik)"
        poin = -1
    else:
        status = f"NETRAL/SEIMBANG ⚪ ({total_berita} Berita Unik)"
        poin = 0
        
    return status, poin

def fetch_intraday_data(period='20d', interval='1h'):
    print("[*] Mengunduh data pasar Bitcoin (Real-time)...")
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
        kurs_idr = float(idr_data['Close'].iloc[-1])
    except:
        kurs_idr = 16000.0

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns and col != 'Volume':
            df[col] = df[col] * kurs_idr
            
    return df

def calculate_advanced_indicators(df):
    print("[*] Menghitung Indikator Teknikal Pro Max...")
    
    # 1. Moving Averages (MA & EMA) - Lama & Baru
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 2. Bollinger Bands (Dipertahankan dari versi awal)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_20 * 2.0)
    df['BB_Lower'] = df['BB_Middle'] - (std_20 * 2.0)
    
    # 3. RSI & Stochastic RSI (Baru)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Menghitung StochRSI (Lebih sensitif)
    min_rsi = df['RSI'].rolling(14).min()
    max_rsi = df['RSI'].rolling(14).max()
    df['StochRSI'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    
    # 4. MACD (Dipertahankan)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 5. ATR (Volatilitas)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # 6. ADX (Average Directional Index - BARU) -> Mengukur Kekuatan Trend
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                         np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['DX'] = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].ewm(alpha=1/14, adjust=False).mean()

    # 7. Volume Trend & Rolling VWAP (BARU)
    if 'Volume' in df.columns:
        df['Vol_MA'] = df['Volume'].rolling(window=24).mean()
        # Perkiraan VWAP 24 Jam Terakhir
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VP'] = df['Typical_Price'] * df['Volume']
        df['VWAP_24'] = df['VP'].rolling(window=24).sum() / df['Volume'].rolling(window=24).sum()
    else:
        df['VWAP_24'] = df['Close'] # Fallback

    return df

def generate_signals_and_predict(df, news_score):
    print("[*] Menganalisis Logika Indikator Gabungan...")
    scores = []
    predictions = []
    
    for i in range(len(df)):
        score = 0
        if i < 200: # Lewati baris yang belum punya data MA200
            scores.append(0)
            predictions.append(np.nan)
            continue
            
        row = df.iloc[i]
        
        # --- LOGIKA LAMA (Dipertahankan) ---
        if row['MA_50'] > row['MA_200']: score += 1
        else: score -= 1
        
        if row['RSI'] < 30: score += 2
        elif row['RSI'] > 70: score -= 2
            
        if row['MACD'] > row['Signal']: score += 1
        else: score -= 1
            
        if row['Close'] <= row['BB_Lower']: score += 1
        elif row['Close'] >= row['BB_Upper']: score -= 1

        # --- LOGIKA BARU (Upgrade) ---
        # EMA Crossover (Sangat peka untuk 1 Jam)
        if row['EMA_20'] > row['EMA_50']: score += 1
        else: score -= 1
            
        # StochRSI (Filter momentum presisi)
        if row['StochRSI'] < 0.2: score += 1 # Oversold tajam
        elif row['StochRSI'] > 0.8: score -= 1 # Overbought tajam
            
        # VWAP (Rata-rata volume harga)
        if row['Close'] > row['VWAP_24']: score += 1
        else: score -= 1

        # ADX (Kekuatan Tren)
        trend_strength = 1.5 if row['ADX'] > 25 else 0.5 # Jika tren kuat, sinyal lebih valid
        score = score * trend_strength
        
        # Tambahkan sentimen berita ke data terakhir
        if i == len(df) - 1:
            score += (news_score * 2) # Berita diberi bobot besar di versi ini

        scores.append(score)
        
        # Menghitung Prediksi Regresi Linear (Machine Learning Lite)
        y = df['Close'].iloc[i-11:i+1].values
        x = np.arange(12)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        predictions.append(p(12)) # Prediksi jam ke-13

    df['Signal_Score'] = scores
    df['Predicted_Next_Close'] = predictions

    # Penilaian kesimpulan yang lebih informatif (Skala lebih luas)
    kondisi = []
    for s in df['Signal_Score']:
        if s >= 6: kondisi.append('🌟 MOMENTUM EMAS UNTUK BELI (STRONG BUY)')
        elif 2 <= s < 6: kondisi.append('📈 POTENSI NAIK, BOLEH BELI (BUY)')
        elif -2 < s < 2: kondisi.append('⚖️ PASAR RAGU-RAGU, TAHAN DULU (HOLD)')
        elif -6 < s <= -2: kondisi.append('📉 POTENSI TURUN, WASPADA (SELL)')
        else: kondisi.append('🚨 BAHAYA! PASAR ANJLOK (STRONG SELL)')
        
    df['Recommendation'] = kondisi
    return df

def evaluate_previous_prediction(df):
    if len(df) < 3: return "Data belum cukup"
    
    prev_signal = df['Recommendation'].iloc[-2]
    prev_price = df['Close'].iloc[-2]
    current_price = df['Close'].iloc[-1]
    
    price_diff = current_price - prev_price
    persentase = (price_diff / prev_price) * 100
    
    # 1. Jika jam lalu sinyalnya BELI (BUY)
    if 'BELI' in prev_signal:
        if current_price > prev_price: 
            return f"BENAR ✅ (Cuan {persentase:.2f}%)"
        else: 
            return f"SALAH ❌ (Meleset {persentase:.2f}%)"
            
    # 2. Jika jam lalu sinyalnya JUAL (SELL)
    elif 'TURUN' in prev_signal or 'ANJLOK' in prev_signal:
        if current_price < prev_price: 
            return f"BENAR ✅ (Berhasil Hindari Minus {persentase:.2f}%)"
        else: 
            return f"SALAH ❌ (Malah Naik {persentase:.2f}%)"
            
    # 3. Jika jam lalu sinyalnya HOLD (Netral)
    else:
        # Jika harga bergerak sangat kecil (di bawah 0.3%), berarti keputusan HOLD memang tepat
        if abs(persentase) <= 0.3: 
            return f"TEPAT ⚪ (Pasar memang sepi: {persentase:.2f}%)"
        # Jika harga naik lumayan tinggi, berarti bot kehilangan momen (Terlewat)
        elif persentase > 0.3:
            return f"TERLEWAT ⚠️ (Sinyal HOLD, tapi harga Naik {persentase:.2f}%)"
        # Jika harga turun tajam, bot gagal memprediksi kejatuhan
        else:
            return f"TERLEWAT ⚠️ (Sinyal HOLD, tapi harga Turun {persentase:.2f}%)"

def plot_professional_analysis(df, filename="chart.png"):
    plot_data = df.tail(80) 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Plot 1: Harga, BB, VWAP, Prediksi
    ax1.plot(plot_data.index, plot_data['Close'], label='Harga BTC', color='black', linewidth=1.5)
    ax1.plot(plot_data.index, plot_data['VWAP_24'], label='VWAP (Garis Imbang Volume)', color='#ff7f0e', linestyle='-.', linewidth=2)
    ax1.plot(plot_data.index, plot_data['MA_50'], label='Trend Menengah (MA50)', color='blue', alpha=0.6)
    
    # Kembalikan Bollinger Bands yang diminta
    ax1.fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], color='gray', alpha=0.15, label='Zona Bollinger')

    last_time = plot_data.index[-1]
    next_time = last_time + pd.Timedelta(hours=1)
    next_price = plot_data['Predicted_Next_Close'].iloc[-1]
    
    ax1.scatter(next_time, next_price, color='red', s=250, marker='*', zorder=10, 
                label=f'Titik Prediksi 1 Jam: {format_rupiah(next_price)}')
    ax1.plot([last_time, next_time], [plot_data['Close'].iloc[-1], next_price], color='red', linestyle=':', linewidth=2)

    ax1.set_title('PRO-MAX QUANT TRADING CHART (1 JAM) - WITA', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Harga (IDR)')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(',', '.')))
    
    myFmt = mdates.DateFormatter('%d %b\n%H:%M')
    
    # Plot 2: Momentum (RSI & StochRSI)
    ax2.plot(plot_data.index, plot_data['RSI'], color='purple', label='RSI (Momentum Harga)')
    ax2.plot(plot_data.index, plot_data['StochRSI']*100, color='cyan', alpha=0.5, label='StochRSI (Sensitif)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.fill_between(plot_data.index, plot_data['RSI'], 70, where=(plot_data['RSI'] >= 70), facecolor='red', alpha=0.3)
    ax2.fill_between(plot_data.index, plot_data['RSI'], 30, where=(plot_data['RSI'] <= 30), facecolor='green', alpha=0.3)
    ax2.set_ylabel('Momentum')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')

    # Plot 3: ADX (Kekuatan Tren)
    ax3.plot(plot_data.index, plot_data['ADX'], color='brown', linewidth=2, label='ADX (Kekuatan Tren)')
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

def send_to_telegram(message, image_path):
    print("[*] Mengirim laporan Pro Max ke Telegram...")
    url_message = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload_msg = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    requests.post(url_message, data=payload_msg)

    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        payload_photo = {'chat_id': TELEGRAM_CHAT_ID}
        requests.post(url_photo, data=payload_photo, files={'photo': photo})
    print("[*] Selesai!")

def main():
    print("==========================================================")
    print("      QUANT TRADING BOT PRO MAX - BITCOIN INDODAX         ")
    print("==========================================================")
    
    df = fetch_intraday_data(period='20d', interval='1h')
    news_status, news_score = fetch_crypto_news_sentiment()
    df = calculate_advanced_indicators(df)
    df = generate_signals_and_predict(df, news_score)

    if len(df) > 2:
        latest = df.iloc[-1]
        akurasi = evaluate_previous_prediction(df)
        
        # Format Laporan Bahasa Manusia (Informative & Easy to Understand)
        pesan = f"💎 *LAPORAN ANALISIS BITCOIN PRO MAX* 💎\n"
        pesan += f"_{latest.name.strftime('%d %B %Y | Pukul %H:%M WITA')}_\n\n"
        
        pesan += f"💰 *Harga Saat Ini:* {format_rupiah(latest['Close'])}\n\n"
        
        pesan += f"🤖 *EVALUASI KINERJA BOT 1 JAM LALU:*\n"
        pesan += f"└ Prediksi sebelumnya terbukti: *{akurasi}*\n\n"
        
        arah_prediksi = "📈 Diprediksi NAIK ke" if latest['Predicted_Next_Close'] > latest['Close'] else "📉 Diprediksi TURUN ke"
        pesan += f"🔮 *TEROPONG HARGA 1 JAM KE DEPAN:*\n"
        pesan += f"└ {arah_prediksi} *{format_rupiah(latest['Predicted_Next_Close'])}*\n\n"
        
        pesan += f"📰 *RADAR BERITA GLOBAL (Anti-Duplikat):*\n"
        pesan += f"└ Sentimen Media: {news_status}\n\n"
        
        # Terjemahan Indikator Teknis ke Bahasa Awam
        tren_adx = "Sedang Trending Kuat 💪" if latest['ADX'] > 25 else "Pasar Sedang Lesu / Sideways 🥱"
        posisi_vwap = "Harga masih di atas rata-rata volume (Bagus) 🟢" if latest['Close'] > latest['VWAP_24'] else "Harga jatuh di bawah rata-rata volume (Waspada) 🔴"
        momentum = "Jenuh Beli (Rentan Turun) 🔴" if latest['RSI'] > 70 else "Jenuh Jual (Peluang Mantul) 🟢" if latest['RSI'] < 30 else "Netral ⚪"
        
        pesan += f"📊 *ISI KEPALA INDIKATOR (Disederhanakan):*\n"
        pesan += f"• Kekuatan Pasar: {tren_adx}\n"
        pesan += f"• Posisi Bandar: {posisi_vwap}\n"
        pesan += f"• Momentum: {momentum}\n\n"
        
        pesan += f"📌 *KESIMPULAN & SARAN AKHIR:*\n"
        pesan += f"*{latest['Recommendation']}*\n\n"
        
        if 'BELI' in latest['Recommendation']:
            sl = latest['Close'] - (latest['ATR'] * 1.5)
            tp = latest['Close'] + (latest['ATR'] * 2.5)
            pesan += f"💡 _Jika Anda memutuskan untuk beli sekarang, pasang Batas Rugi (Stop Loss) di angka {format_rupiah(sl)} dan amankan untung (Take Profit) sekitar {format_rupiah(tp)}._\n"
            
        pesan += "\n_ℹ️ Disclaimer: Bot menganalisa data matematis dan berita masa lalu, tidak menjamin kepastian masa depan 100%._"

        chart_filename = "promax_chart.png"
        plot_professional_analysis(df, chart_filename)
        send_to_telegram(pesan, chart_filename)
    else:
        print("[!] Gagal memproses data.")

if __name__ == "__main__":
    main()
