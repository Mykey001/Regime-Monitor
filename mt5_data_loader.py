"""
MT5 Data Loader for Regime Analysis
====================================

Fetches and processes real-time data from MetaTrader 5.
Prepares data for regime model prediction.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MT5DataLoader:
    """Load and process MT5 data for regime analysis."""
    
    def __init__(self, symbol='XAUUSDm', timeframe=mt5.TIMEFRAME_M15):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initialized = False
        
    def initialize(self):
        """Initialize MT5 connection."""
        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False
        
        # Check symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found")
            return False
        
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select {self.symbol}")
                return False
        
        self.initialized = True
        logger.info(f"MT5 initialized for {self.symbol}")
        return True
    
    def shutdown(self):
        """Shutdown MT5 connection."""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            logger.info("MT5 connection closed")
    
    def get_rates(self, n_bars=500):
        """
        Fetch historical rates from MT5.
        
        Args:
            n_bars: Number of bars to fetch (need 200+ for indicators)
        
        Returns:
            DataFrame with OHLCV data or None on error
        """
        if not self.initialized:
            if not self.initialize():
                return None
        
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, n_bars)
        
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns to match model expectations
        df = df.rename(columns={
            'tick_volume': 'tickvol'
        })
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'tickvol']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return None
        
        # Add spread column if missing (model expects it)
        if 'spread' not in df.columns:
            df['spread'] = np.nan
        
        return df[['open', 'high', 'low', 'close', 'tickvol', 'spread']]
    
    def get_current_tick(self):
        """Get current tick data."""
        if not self.initialized:
            if not self.initialize():
                return None
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error(f"Failed to get tick: {mt5.last_error()}")
            return None
        
        return {
            'time': datetime.fromtimestamp(tick.time),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume
        }
    
    def get_symbol_info(self):
        """Get symbol information."""
        if not self.initialized:
            if not self.initialize():
                return None
        
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return None
        
        return {
            'description': info.description,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'trade_contract_size': info.trade_contract_size,
            'currency_base': info.currency_base,
            'currency_profit': info.currency_profit,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step
        }
    
    def timeframe_to_string(self):
        """Convert MT5 timeframe to string."""
        timeframes = {
            mt5.TIMEFRAME_M1: "M1",
            mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1",
            mt5.TIMEFRAME_W1: "W1",
            mt5.TIMEFRAME_MN1: "MN1"
        }
        return timeframes.get(self.timeframe, "Unknown")
    
    def timeframe_to_minutes(self):
        """Convert MT5 timeframe to minutes."""
        minutes = {
            mt5.TIMEFRAME_M1: 1,
            mt5.TIMEFRAME_M5: 5,
            mt5.TIMEFRAME_M15: 15,
            mt5.TIMEFRAME_M30: 30,
            mt5.TIMEFRAME_H1: 60,
            mt5.TIMEFRAME_H4: 240,
            mt5.TIMEFRAME_D1: 1440,
            mt5.TIMEFRAME_W1: 10080,
            mt5.TIMEFRAME_MN1: 43200
        }
        return minutes.get(self.timeframe, 15)


def test_connection():
    """Test MT5 connection and data retrieval."""
    print("Testing MT5 Connection...")
    print("=" * 50)
    
    loader = MT5DataLoader(symbol='XAUUSDm', timeframe=mt5.TIMEFRAME_M15)
    
    if not loader.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    print("✓ MT5 initialized")
    
    # Get symbol info
    info = loader.get_symbol_info()
    if info:
        print(f"\n✓ Symbol Info:")
        print(f"  Description: {info['description']}")
        print(f"  Digits: {info['digits']}")
        print(f"  Point: {info['point']}")
        print(f"  Spread: {info['spread']}")
    
    # Get current tick
    tick = loader.get_current_tick()
    if tick:
        print(f"\n✓ Current Tick:")
        print(f"  Time: {tick['time']}")
        print(f"  Bid: {tick['bid']:.5f}")
        print(f"  Ask: {tick['ask']:.5f}")
    
    # Get historical data
    df = loader.get_rates(n_bars=100)
    if df is not None:
        print(f"\n✓ Historical Data:")
        print(f"  Bars: {len(df)}")
        print(f"  Timeframe: {loader.timeframe_to_string()}")
        print(f"  Date Range: {df.index.min()} to {df.index.max()}")
        print(f"\n  Latest bars:")
        print(df.tail(3))
    else:
        print("\n❌ Failed to get historical data")
    
    loader.shutdown()
    print("\n" + "=" * 50)
    print("Test complete")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_connection()
