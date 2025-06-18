from typing import Optional, Any, Tuple, Dict, cast
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt
import pandas as pd

ObsType = npt.NDArray[np.float32]
ActType = int

class TradingEnv(gym.Env[ObsType, ActType]):
    """Trading environment for reinforcement learning."""
    
    metadata = {"render_modes": [], "render_fps": 30}
    
    def __init__(self, data: pd.DataFrame, lstm_preds: npt.NDArray[np.float32], prophet_preds: npt.NDArray[np.float32], tp_pct: float = 0.02, sl_pct: float = 0.01) -> None:
        """Initialize the environment."""
        super().__init__()
        self.data: pd.DataFrame = data
        self.lstm_preds: npt.NDArray[np.float32] = lstm_preds.astype(np.float32)
        self.prophet_preds: npt.NDArray[np.float32] = prophet_preds.astype(np.float32)
        self.current_step: int = 0
        
        # Create action and observation spaces
        self.action_space: spaces.Space[ActType] = cast(spaces.Space[ActType], spaces.Discrete(3))  # 0: hold, 1: buy, 2: sell
        self.observation_space: spaces.Space[ObsType] = cast(spaces.Space[ObsType], spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(3,), 
            dtype=np.float32
        ))
        
        self.position: Optional[str] = None  # None, 'long'
        self.entry_price: float = 0.0
        self.tp_pct: float = tp_pct
        self.sl_pct: float = sl_pct
        
        # Validate data
        if 'close' not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")

    def _get_close_price(self, step: int) -> float:
        """Safely get close price from DataFrame."""
        try:
            return float(self.data['close'].iloc[step])
        except (IndexError, KeyError):
            return 0.0001

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            super().reset(seed=seed, options=options)
        self.current_step = 0
        self.position = None
        self.entry_price = 0.0
        return self._get_obs(), {}

    def _get_obs(self) -> ObsType:
        """Get the current observation."""
        try:
            current_price = self._get_close_price(self.current_step)
            lstm_pred = float(self.lstm_preds[self.current_step])
            prophet_pred = float(self.prophet_preds[self.current_step])
            
            # Handle NaN or infinite values
            if np.isnan(current_price) or np.isinf(current_price):
                current_price = self._get_close_price(max(0, self.current_step-1)) if self.current_step > 0 else 0.0001
            
            if np.isnan(lstm_pred) or np.isinf(lstm_pred):
                lstm_pred = current_price  # Use current price if prediction is invalid
            
            if np.isnan(prophet_pred) or np.isinf(prophet_pred):
                prophet_pred = current_price  # Use current price if prediction is invalid
            
            # Create observation with validated values
            obs = np.array([current_price, lstm_pred, prophet_pred], dtype=np.float32)
            
            # Final validation to ensure no NaN values
            if np.isnan(obs).any() or np.isinf(obs).any():
                obs = np.array([current_price, current_price, current_price], dtype=np.float32)
            
            # Normalize values to prevent numerical instability
            price_scale = max(abs(current_price), 1e-8)  # Avoid division by zero
            return obs / price_scale
            
        except Exception as e:
            print(f"Error in _get_obs: {str(e)}")
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        try:
            reward: float = 0.0
            price: float = self._get_close_price(self.current_step)
            
            # Validate price
            if np.isnan(price) or np.isinf(price):
                price = self._get_close_price(max(0, self.current_step-1)) if self.current_step > 0 else 0.0001
            
            done: bool = False
            info: Dict[str, Any] = {}

            # Entry logic
            if self.position is None:
                if action == 1:  # Buy
                    self.position = 'long'
                    self.entry_price = price
            else:
                # Check TP/SL
                if self.position == 'long':
                    tp_price = self.entry_price * (1 + self.tp_pct)
                    sl_price = self.entry_price * (1 - self.sl_pct)
                    
                    # Calculate rewards
                    if price >= tp_price:
                        reward = min(tp_price - self.entry_price, tp_price * 0.1)  # Cap reward at 10%
                        self.position = None
                        self.entry_price = 0.0
                    elif price <= sl_price:
                        reward = max(sl_price - self.entry_price, -sl_price * 0.1)  # Cap loss at -10%
                        self.position = None
                        self.entry_price = 0.0
                    elif action == 2:  # Manual close (sell)
                        reward = np.clip(price - self.entry_price, -price * 0.1, price * 0.1)  # Cap at ±10%
                        self.position = None
                        self.entry_price = 0.0

            self.current_step += 1
            if self.current_step >= len(self.data) - 1:
                done = True
                # Close open position at end
                if self.position == 'long':
                    reward += self._get_close_price(self.current_step) - self.entry_price
                    self.position = None
                    self.entry_price = 0.0

            obs = self._get_obs() if not done else np.zeros(3, dtype=np.float32)
            return obs, reward, done, False, info
            
        except Exception as e:
            print(f"Error in step: {str(e)}")
            return np.zeros(3, dtype=np.float32), 0.0, True, False, {}
