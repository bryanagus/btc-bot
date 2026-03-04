🚀 BTC Quant Godmode Pro Max
BTC Quant Godmode Pro Max adalah sistem algorithmic trading cerdas untuk Bitcoin (BTC/IDR) di Indodax. Sistem ini menggabungkan analisis teknikal, Ensemble Machine Learning, simulasi stokastik (Monte Carlo), dan pemrosesan bahasa alami (NLP) untuk memberikan rekomendasi perdagangan dengan tingkat akurasi yang diuji secara ketat tanpa data leakage.
Sistem ini dirancang untuk berjalan secara serverless 24/7 menggunakan GitHub Actions dan mengirimkan laporan lengkap beserta grafik ke Telegram Anda.
✨ Fitur Utama
 * 🧠 Ensemble Machine Learning Anti-Leakage: Menggabungkan RandomForest, GradientBoosting, dan LogisticRegression. Divalidasi secara ketat menggunakan TimeSeriesSplit agar AI tidak "mengintip" data masa depan.
 * ⚡ Real-Time Indodax API: Tidak ada delay. Prediksi AI selalu menggunakan harga detik terakhir langsung dari public API Indodax.
 * 📰 VADER NLP News Sentiment: Membaca dan menganalisis emosi dari puluhan berita kripto global terbaru (CoinDesk, CoinTelegraph, dll) untuk mendeteksi sentimen bullish atau bearish.
 * 🎲 Monte Carlo & Dynamic Kelly Criterion: Mengkalkulasi Value at Risk (VaR 95%) dan mensimulasikan ribuan skenario harga 24 jam ke depan untuk memberikan rekomendasi persentase alokasi modal (Risk/Reward dinamis).
 * 📊 Dual Auto-Charting: Menghasilkan dua grafik profesional secara otomatis:
   * Main Chart (80 Jam): Lengkap dengan VWAP, MA, Bollinger Bands, RSI, dan ADX.
   * Zoom Chart (6 Jam): Fokus pada pergerakan jangka pendek beserta titik target AI.
 * 📲 Telegram Integration: Laporan komprehensif dikirim langsung ke grup/chat pribadi Telegram Anda.
📸 Pratinjau Laporan Telegram
(Tambahkan screenshot laporan bot Telegram kamu di sini)
     
🛠️ Instalasi & Pengaturan
Sistem ini didesain untuk berjalan otomatis menggunakan GitHub Actions yang dipicu (triggered) melalui layanan cron gratis seperti cron-job.org.
1. Persiapan Repositori
 * Fork atau buat repositori baru di GitHub Anda.
 * Pastikan file main.py dan requirements.txt sudah ter-upload di repositori tersebut.
2. Pengaturan Rahasia (GitHub Secrets)
Bot memerlukan token Telegram agar bisa mengirim pesan.
 * Buka Repositori GitHub Anda > Settings > Secrets and variables > Actions.
 * Klik New repository secret dan tambahkan 2 rahasia berikut:
   * TELEGRAM_BOT_TOKEN: Isi dengan token bot Telegram Anda (dapatkan dari BotFather).
   * TELEGRAM_CHAT_ID: Isi dengan ID Chat Anda atau Grup Anda.
3. Setup GitHub Actions
Buat file workflow untuk menjalankan bot secara otomatis.
 * Di repositori Anda, buat folder .github/workflows/.
 * Di dalam folder tersebut, buat file bernama bot.yml.
 * Isi dengan kode berikut:
<!-- end list -->
name: Run Quant Godmode Bot

on:
  workflow_dispatch:
  repository_dispatch:
    types: [run-bot]

jobs:
  trade-bot:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run Godmode Engine
      env:
        TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
        TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      run: python main.py

4. Otomatisasi dengan Cron-Job.org
Agar script berjalan setiap 1 jam, atur trigger eksternal:
 * Buat Personal Access Token (PAT) di akun GitHub Anda (beri akses repo).
 * Daftar/Login ke cron-job.org.
 * Buat cronjob baru dengan jadwal setiap 1 jam (atau sesuai keinginan).
 * Arahkan URL ke: https://api.github.com/repos/USERNAME_KAMU/NAMA_REPO_KAMU/dispatches
 * Gunakan metode POST.
 * Tambahkan Header:
   * Accept: application/vnd.github.v3+json
   * Authorization: token ISI_DENGAN_PAT_KAMU
 * Isi Body dengan: {"event_type": "run-bot"}
📦 Dependencies (requirements.txt)
Pastikan file requirements.txt Anda memiliki library berikut:
pandas==2.2.1
numpy==1.26.4
matplotlib==3.8.3
pytz==2024.1
requests==2.31.0
yfinance==0.2.37
scikit-learn==1.4.1.post1
vaderSentiment==3.3.2

⚠️ Disclaimer Peringatan Risiko
> Perhatian: Skrip ini murni merupakan alat bantu komputasi dan bukan penasihat keuangan. Perdagangan mata uang kripto (Cryptocurrency) memiliki tingkat risiko yang sangat tinggi dan dapat mengakibatkan hilangnya sebagian atau seluruh modal Anda. Segala keputusan jual/beli yang dilakukan berdasarkan sinyal dari bot ini adalah tanggung jawab pengguna sepenuhnya. Penulis tidak bertanggung jawab atas kerugian finansial apa pun yang mungkin terjadi. Gunakan dengan bijak.
> 
Dibuat dengan ❤️ untuk komunitas Algo-Trading.
