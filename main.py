from data_loader import get_historical_klines
from lstm_model import build_lstm_model
from prophet_model import train_prophet
from rl_agent import TradingEnv
from stable_baselines3 import PPO
from typing import List, Optional, Dict, Any
import numpy as np
import requests
import os
import pandas as pd
import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *
import sys

def setup_logger(log_dir: str = 'logs') -> str:
    """Mengatur konfigurasi logging untuk mencatat aktivitas bot"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )
    return log_path

def run_multi_coin(
    symbols: List[str],
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    source: str = 'binance',
    coingecko_ids: Optional[List[str]] = None,
    log_dir: str = 'logs',
    auto_tune: bool = False,
    test_mode: bool = True
) -> Dict[str, Any]:
    """
    Run trading bot for multiple coins simultaneously.
    
    Args:
        symbols: List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        tp_pct: Take Profit percentage (default: 0.02 or 2%)
        sl_pct: Stop Loss percentage (default: 0.01 or 1%)
        source: Data source ('binance' or 'coingecko')
        coingecko_ids: List of CoinGecko IDs when using CoinGecko as source
        log_dir: Directory to store log files
        auto_tune: Whether to use hyperparameter auto-tuning
        test_mode: Whether to run in test mode (default: True)
          Returns:
        Dict[str, Any]: Trading results for each symbol
            - symbol: Dict containing model predictions and actions
            - performance metrics
            - trading history
            
    Raises:
        ValueError: If invalid source or parameters are provided
        RuntimeError: If there are issues with data fetching or model training
    """
    import optuna
    
    # Validate inputs
    if not symbols:
        raise ValueError("No trading symbols provided")
    if not (0 < tp_pct < 1):
        raise ValueError(f"Invalid take profit percentage: {tp_pct}. Must be between 0 and 1")
    if not (0 < sl_pct < 1):
        raise ValueError(f"Invalid stop loss percentage: {sl_pct}. Must be between 0 and 1")
    if source not in ['binance', 'coingecko']:
        raise ValueError(f"Invalid data source: {source}. Must be 'binance' or 'coingecko'")
    if source == 'coingecko' and (not coingecko_ids or len(coingecko_ids) != len(symbols)):
        raise ValueError("Must provide CoinGecko IDs for each symbol when using CoinGecko as source")
    
    # Setup logging
    try:
        os.makedirs(log_dir, exist_ok=True)
        logger_path = setup_logger(log_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to setup logging directory: {str(e)}")
    
    logging.info(f'Starting multi-coin trading. Parameters: TP={tp_pct:.2%}, SL={sl_pct:.2%}, source={source}, auto_tune={auto_tune}')

    def tune_objective(trial: optuna.Trial, df: pd.DataFrame, prophet_preds: np.ndarray) -> float:
        """
        Optimize model hyperparameters using Optuna.
        
        Args:
            trial: Optuna trial object
            df: DataFrame containing historical data
            prophet_preds: Prophet model predictions
            
        Returns:
            float: Negative mean squared error (for maximization)
        """
        window = trial.suggest_int('window', 30, 120)
        units = trial.suggest_int('units', 32, 128)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        X, y = [], []
        for i in range(window, len(df)):
            X.append(df['close'].values[i-window:i])
            y.append(df['close'].values[i])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        if len(X) == 0:
            logging.warning("Not enough data points for the given window size")
            return float('inf')  # Return worst possible score
            
        try:
            model = build_lstm_model((window, 1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=2, batch_size=32, verbose=0)
            preds = model.predict(X, verbose=0).flatten()
            mse = np.mean((preds - y) ** 2)
            return -mse
        except Exception as e:
            logging.error(f"Error in hyperparameter optimization: {str(e)}")
            return float('inf')  # Return worst possible score

    results = {}
    for idx, symbol in enumerate(symbols):
        logging.info(f'\n=== Processing {symbol} ===')
        try:
            # Fetch historical data
            if source == 'binance':
                df = get_historical_klines(symbol=symbol)
            else:  # source == 'coingecko'
                df = get_historical_ohlc_coingecko(coingecko_ids[idx], days=30)
                
            if df is None or df.empty:
                raise RuntimeError(f"No data received for {symbol}")
                
            # Add technical indicators
            df = add_technical_indicators(df)
            
            # Train Prophet model
            try:
                prophet_model = train_prophet(df)
                future = prophet_model.make_future_dataframe(periods=0, freq='H')
                forecast = prophet_model.predict(future)
                prophet_preds = forecast['yhat'].values
            except Exception as e:
                raise RuntimeError(f"Prophet model training failed for {symbol}: {str(e)}")
                
            results[symbol] = {
                'data': df,
                'prophet_preds': prophet_preds
            }            # Multi-strategy: LSTM, Prophet, MA20
            window = 60
            X, y = [], []
            try:
                # Prepare LSTM data
                for i in range(window, len(df)):
                    X.append(df['close'].values[i-window:i])
                    y.append(df['close'].values[i])
                X, y = np.array(X), np.array(y)
                if len(X) < window:
                    raise ValueError(f"Insufficient data points for LSTM model (got {len(X)}, need at least {window})")
                X = X.reshape((X.shape[0], X.shape[1], 1))
                # Train LSTM model
                lstm_model = build_lstm_model((window, 1))
                lstm_model.compile(optimizer='adam', loss='mse')
                history = lstm_model.fit(X, y, epochs=2, batch_size=32, verbose=0, validation_split=0.2)
                # Generate predictions
                lstm_preds = np.concatenate([
                    np.zeros(window),  # Padding for initial window
                    lstm_model.predict(X, verbose=0).flatten()
                ])
                results[symbol]['lstm'] = {
                    'predictions': lstm_preds,
                    'history': history.history,
                    'window': window
                }
            except Exception as e:
                logging.error(f"LSTM model training failed for {symbol}: {str(e)}")
                lstm_preds = np.zeros(len(df))
                results[symbol]['lstm'] = {
                    'predictions': lstm_preds,
                    'error': str(e)
                }

            # Pastikan kode di bawah ini berada di luar blok try/except
            ma20_preds = df['MA20'].values
            predictions_dict = {
                'LSTM': lstm_preds,
                'Prophet': prophet_preds,
                'MA20': ma20_preds
            }

            # Eksekusi trading real berdasarkan sinyal RL untuk setiap strategi
            from binance.client import Client
            API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_API_KEY')
            API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_API_SECRET')
            client = Client(API_KEY, API_SECRET)
            qty = 0.001  # Atur sesuai kebutuhan
            for strat_name, strat_preds in predictions_dict.items():
                env = TradingEnv(df, strat_preds, prophet_preds, tp_pct=tp_pct, sl_pct=sl_pct)
                model = PPO('MlpPolicy', env, verbose=0)

                # model.learn(total_timesteps=10000)
                # Initialize variables before the loop
                rewards = []
                actions = []
                obs, _ = env.reset()
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, _ = env.step(action)
                    rewards.append(reward)
                    actions.append(action)
                
                log_path = os.path.join(log_dir, f'{symbol}_{strat_name}_reward_log.csv')
                pd.DataFrame({'reward': rewards, 'action': actions}).to_csv(log_path, index=False)
                logging.info(f'Log reward RL {strat_name} telah disimpan di {log_path}')
                print(f'Log reward RL {strat_name} telah disimpan di {log_path}')
                
                # Eksekusi order langsung (hanya aksi beli/jual, lewati tahan)
                for act in actions:
                    if act == 1:  # BUY
                        order = place_order(client, symbol, 'BUY', qty)
                        if order:
                            logging.info(f'Order BUY {symbol} success: {order}')
                            print(f'Order BUY {symbol} success: {order}')
                    elif act == 2:  # SELL
                        order = place_order(client, symbol, 'SELL', qty)
                        if order:
                            logging.info(f'Order SELL {symbol} success: {order}')
                            print(f'Order SELL {symbol} success: {order}')
        except Exception as e:
            logging.error(f'Error processing {symbol}: {str(e)}')
            print(f'Error processing {symbol}: {str(e)}')
        print(f'Pipeline {symbol} selesai.\n')

def get_trending_symbols(limit=5):
    """Mengambil simbol crypto yang sedang ramai diperdagangkan di Binance (berdasarkan volume 24 jam)."""
    url = 'https://api.binance.com/api/v3/ticker/24hr'
    response = requests.get(url)
    data = response.json()
    # Filter hanya pasangan USDT
    usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
    # Urutkan berdasarkan volume quote (USDT)
    usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
    trending = [item['symbol'] for item in usdt_pairs[:limit]]
    return trending

def run_multi_coin_auto(tp_pct=0.02, sl_pct=0.01, limit=5, source='binance', test_mode=True):
    """
    Menjalankan bot trading otomatis untuk coin-coin trending
    
    Parameters:
    - tp_pct: Persentase Take Profit (default: 0.02)
    - sl_pct: Persentase Stop Loss (default: 0.01)
    - limit: Jumlah coin yang akan diproses (default: 5)
    - source: Sumber data ('binance' atau 'coingecko')
    - test_mode: Mode test untuk keamanan trading (default: True)
    """
    if source == 'binance':
        trending_symbols = get_trending_symbols(limit=limit)
        print(f"Simbol trending: {trending_symbols}")
        run_multi_coin(trending_symbols, tp_pct=tp_pct, sl_pct=sl_pct, source='binance', test_mode=test_mode)
    elif source == 'coingecko':
        # Ambil top coin dari Coingecko
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {'vs_currency': 'usd', 'order': 'volume_desc', 'per_page': limit, 'page': 1}
        response = requests.get(url, params=params)
        data = response.json()
        ids = [item['id'] for item in data]
        symbols = [item['symbol'].upper()+'USDT' for item in data]
        print(f"ID Trending Coingecko: {ids}")
        run_multi_coin(symbols, tp_pct=tp_pct, sl_pct=sl_pct, source='coingecko', coingecko_ids=ids)

def evaluate_rl_performance(reward_log_path: str) -> Dict[str, float]:
    """
    Evaluate performance of the RL model from log file.
    
    Args:
        reward_log_path: Path to the reward log CSV file
        
    Returns:
        Dict[str, float]: Dictionary containing performance metrics
            - total_reward: Total cumulative reward
            - mean_reward: Average reward per step
            - max_reward: Maximum reward achieved
            - min_reward: Minimum reward received
    """
    import pandas as pd
    import numpy as np
    
    if not os.path.exists(reward_log_path):
        logging.error(f"Reward log file not found: {reward_log_path}")
        return {}
    
    df = pd.read_csv(reward_log_path)
    total_reward = df['reward'].sum()
    mean_reward = df['reward'].mean()
    max_reward = df['reward'].max()
    min_reward = df['reward'].min()
    metrics = {
        'total_reward': float(total_reward),
        'mean_reward': float(mean_reward),
        'max_reward': float(max_reward),
        'min_reward': float(min_reward)
    }
    
    logging.info(f"RL Evaluation results from {reward_log_path}:")
    for metric, value in metrics.items():
        logging.info(f"  {metric}: {value:.4f}")
        
    return metrics


def get_binance_client(leverage: int = 5, margin_type: str = 'ISOLATED'):
    """
    Initialize Binance client with futures trading setup
    Returns a configured Binance client instance
    """
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        logging.error("Binance API keys not found in environment variables")
        print("Please set your Binance API keys using:")
        print("$env:BINANCE_API_KEY='your_api_key'")
        print("$env:BINANCE_API_SECRET='your_api_secret'")
        sys.exit(1)
    
    try:
        client = Client(api_key, api_secret)
        # Test Futures API connection
        client.futures_account()
        
        # Setup futures trading untuk semua simbol trending
        for symbol in get_trending_symbols():
            setup_futures_trading(client, symbol, leverage, margin_type)
            
        return client
    except BinanceAPIException as e:
        logging.error(f"Binance API Error: {str(e)}")
        print(f"Error connecting to Binance API: {str(e)}")
        sys.exit(1)

def get_symbol_info(client, symbol):
    """Get futures trading rules for a symbol"""
    try:
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if symbol_info:
            filters = {f['filterType']: f for f in symbol_info['filters']}
            return {
                'minNotional': float(filters['MIN_NOTIONAL']['notional']),
                'minQty': float(filters['LOT_SIZE']['minQty']),
                'stepSize': float(filters['LOT_SIZE']['stepSize'])
            }
    except Exception as e:
        logging.error(f"Error getting symbol info: {str(e)}")
    return None

def calculate_quantity(price, min_notional, min_qty, step_size):
    """Calculate valid quantity based on price and trading rules"""
    # Calculate base quantity from min_notional
    quantity = max(min_notional / price, min_qty)
    
    # Round to step size
    step_size_decimals = len(str(step_size).split('.')[-1])
    quantity = round(quantity - (quantity % step_size), step_size_decimals)
    
    return quantity

def place_order(client, symbol: str, side: str, quantity: float, tp_pct: float = 0.03, sl_pct: float = 0.015):
    """Place futures order with proper quantity validation and TP/SL"""
    try:
        # Get symbol trading rules
        symbol_info = get_symbol_info(client, symbol)
        if not symbol_info:
            logging.error(f"Could not get trading rules for {symbol}")
            return None
            
        # Get current price
        ticker = client.futures_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
          # Calculate proper quantity
        valid_quantity = calculate_quantity(
            current_price,
            symbol_info['minNotional'],
            symbol_info['minQty'],
            symbol_info['stepSize']
        )
        # Close existing position if any
        close_futures_position(client, symbol)
        
        # Place new position with TP/SL
        orders = place_futures_order_with_tp_sl(
            client, 
            symbol, 
            side, 
            valid_quantity, 
            tp_pct, 
            sl_pct
        )
        logging.info(f"Order {side} {symbol} placed successfully: {orders}")
        return orders
        
    except BinanceAPIException as e:
        logging.error(f"Order {side} {symbol} error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error placing order: {str(e)}")
        return None

def setup_futures_trading(client, symbol: str, leverage: int = 5, margin_type: str = 'ISOLATED'):
    """Setup futures trading configuration untuk sebuah simbol

    Parameters:
    - client: Binance client instance
    - symbol: Trading symbol (e.g., 'BTCUSDT')
    - leverage: Leverage yang diinginkan (1-125 tergantung pair)
    - margin_type: 'ISOLATED' atau 'CROSSED'
    """
    try:
        # Set margin type
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
        logging.info(f"Set margin type {margin_type} for {symbol}")
    except BinanceAPIException as e:
        if 'No need to change margin type' not in str(e):
            logging.error(f"Error setting margin type for {symbol}: {e}")
    
    try:
        # Set leverage
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        logging.info(f"Set leverage {leverage}x for {symbol}")
    except BinanceAPIException as e:
        logging.error(f"Error setting leverage for {symbol}: {e}")

    # Get position information
    try:
        position_info = client.futures_position_information(symbol=symbol)
        logging.info(f"Position info for {symbol}: {position_info}")
        return True
    except Exception as e:
        logging.error(f"Error getting position info: {e}")
        return False

def close_futures_position(client, symbol: str):
    """Menutup posisi futures yang ada"""
    try:
        position_info = client.futures_position_information(symbol=symbol)
        for position in position_info:
            if float(position['positionAmt']) != 0:  # Ada posisi terbuka
                side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
                quantity = abs(float(position['positionAmt']))
                
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity,
                    reduceOnly=True
                )
                logging.info(f"Closed position for {symbol}: {order}")
                return True
    except Exception as e:
        logging.error(f"Error closing position: {e}")
    return False

def place_futures_order_with_tp_sl(client, symbol: str, side: str, quantity: float, tp_pct: float = 0.03, sl_pct: float = 0.015):
    """
    Place futures order dengan Take Profit dan Stop Loss
    
    Parameters:
    - client: Binance client instance
    - symbol: Trading symbol
    - side: 'BUY' atau 'SELL'
    - quantity: Jumlah untuk trading
    - tp_pct: Take profit percentage
    - sl_pct: Stop loss percentage
    """
    try:
        # Get current price
        price_info = client.futures_symbol_ticker(symbol=symbol)
        current_price = float(price_info['price'])
        
        # Calculate TP and SL prices
        if side == 'BUY':
            tp_price = current_price * (1 + tp_pct)
            sl_price = current_price * (1 - sl_pct)
        else:  # SELL
            tp_price = current_price * (1 - tp_pct)
            sl_price = current_price * (1 + sl_pct)
        
        # Place main order
        main_order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        
        # Place take profit order
        tp_order = client.futures_create_order(
            symbol=symbol,
            side='SELL' if side == 'BUY' else 'BUY',
            type='TAKE_PROFIT_MARKET',
            stopPrice=tp_price,
            closePosition=True,
            workingType='MARK_PRICE'
        )
        
        # Place stop loss order
        sl_order = client.futures_create_order(
            symbol=symbol,
            side='SELL' if side == 'BUY' else 'BUY',
            type='STOP_MARKET',
            stopPrice=sl_price,
            closePosition=True,
            workingType='MARK_PRICE'
        )
        
        logging.info(f"Orders placed for {symbol}:")
        logging.info(f"Main order: {main_order}")
        logging.info(f"TP order: {tp_order}")
        logging.info(f"SL order: {sl_order}")
        
        return {
            'main_order': main_order,
            'tp_order': tp_order,
            'sl_order': sl_order
        }
        
    except Exception as e:
        logging.error(f"Error placing futures orders: {e}")
        return None

def add_technical_indicators(df):
    df = df.copy()
    # Moving Average
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df
# Contoh integrasi ke pipeline RL:
def rl_trading_execution(symbol, actions, test=True):
    client = get_binance_client()
    qty = 0.001  # Atur sesuai kebutuhan
    for action in actions:
        if action == 1:  # Buy
            execute_binance_trade(client, symbol, 'BUY', qty, test=test)
        elif action == 2:  # Sell
            execute_binance_trade(client, symbol, 'SELL', qty, test=test)

def execute_binance_trade(client, symbol: str, side: str, quantity: float, test: bool = True):
    """
    Execute trade on Binance with improved error handling and validation
    
    Parameters:
    - client: Binance client instance
    - symbol: Trading symbol (e.g., 'BTCUSDT')
    - side: 'BUY' or 'SELL'
    - quantity: Amount to trade
    - test: If True, use test order for safety
    
    Returns:
    - dict: Order details if successful
    - None: If order fails
    """
    try:
        # Validate inputs
        if side not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid side parameter: {side}. Must be 'BUY' or 'SELL'")
        
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}. Must be greater than 0")
            
        # Get symbol info for validation
        symbol_info = get_symbol_info(client, symbol)
        if not symbol_info:
            raise ValueError(f"Could not get trading info for symbol: {symbol}")
            
        # Create order parameters
        order_params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity
        }
        
        # Execute order based on test mode
        if test:
            order = client.create_test_order(**order_params)
            logging.info(f"Test {side} order simulated for {symbol}: {order}")
        else:
            order = client.create_order(**order_params)
            logging.info(f"Live {side} order executed for {symbol}: {order}")
        
        return order
        
    except BinanceAPIException as e:
        error_msg = f"Binance API error executing {side} order for {symbol}: {str(e)}"
        logging.error(error_msg)
        print(error_msg)  # Print for immediate visibility
        return None
        
    except ValueError as e:
        error_msg = f"Validation error: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return None
        
    except Exception as e:
        error_msg = f"Unexpected error executing {side} order for {symbol}: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return None


if __name__ == '__main__':
    # Bot trading otomatis untuk 10 coin trending dari Binance
    # Default menggunakan test_mode=True untuk keamanan
    run_multi_coin_auto(tp_pct=0.03, sl_pct=0.015, limit=10, source='binance', test_mode=True)
    
    # Untuk menggunakan data dari Coingecko (dinonaktifkan):
    # run_multi_coin_auto(tp_pct=0.03, sl_pct=0.015, limit=10, source='coingecko')
    
    # Evaluasi hasil trading untuk setiap coin
    log_dir = 'logs'
    for file in os.listdir(log_dir):
        if file.endswith('_reward_log.csv'):
            evaluate_rl_performance(os.path.join(log_dir, file))
