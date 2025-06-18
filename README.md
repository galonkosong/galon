# Binance AI Trading Bot

Proyek ini menggabungkan LSTM, Prophet, dan Reinforcement Learning untuk membuat bot trading crypto pintar.

## Struktur Proyek
- data_loader.py: Mengambil data historis dari Binance.
- lstm_model.py: Model LSTM untuk prediksi harga.
- prophet_model.py: Model Prophet untuk prediksi harga.
- rl_agent.py: Agent RL (PPO) untuk aksi trading.
- main.py: Integrasi pipeline dan eksekusi bot.

## Instalasi
Jalankan perintah berikut di PowerShell:

pip install -r requirements.txt

## Dependensi Utama
- tensorflow
- keras
- prophet
- stable-baselines3
- python-binance
- pandas, numpy, scikit-learn

## Catatan
- Pastikan sudah memiliki API key Binance untuk trading live.
- Contoh kode awal tersedia di setiap file utama.
