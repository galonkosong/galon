import requests
import pandas as pd
from binance.client import Client

# Get API keys from environment variables
import os
import sys

API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Validate API keys
if not API_KEY or not API_SECRET:
    print("Error: Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
    print("Example:")
    print("$env:BINANCE_API_KEY='your_api_key_here'")
    print("$env:BINANCE_API_SECRET='your_api_secret_here'")
    sys.exit(1)

try:
    client = Client(API_KEY, API_SECRET)
    # Test API connection
    client.get_account()
except Exception as e:
    print(f"Error connecting to Binance API: {str(e)}")
    sys.exit(1)
    print("Warning: Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")

def get_historical_klines(symbol='BTCUSDT', interval='1h', lookback='30 days ago UTC'):
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df[['open_time', 'close']]

def get_historical_ohlc_coingecko(symbol_id, vs_currency='usd', days=30, interval='hourly'):
    """
    Ambil data OHLC historis dari Coingecko.
    symbol_id: id coin di Coingecko (misal: 'bitcoin', 'ethereum')
    vs_currency: 'usd', 'idr', dst
    days: rentang hari ke belakang (1,7,14,30,90,180,365,max)
    interval: 'hourly' atau 'daily'
    """
    url = f'https://api.coingecko.com/api/v3/coins/{symbol_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days, 'interval': interval}
    r = requests.get(url, params=params)
    data = r.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['open_time', 'close'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df[['open_time', 'close']]

if __name__ == '__main__':
    df_binance = get_historical_klines()
    print("Binance Data:")
    print(df_binance.head())

    df_coingecko = get_historical_ohlc_coingecko('bitcoin', days=30)
    print("Coingecko Data:")
    print(df_coingecko.head())
