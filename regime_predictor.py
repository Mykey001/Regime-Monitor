"""
Regime Predictor
================

Loads the trained regime model and provides prediction interface.
Uses the exact feature engineering from the training notebook.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RegimePredictor:
    """Predict market regime using trained model."""
    
    # Regime state names
    REGIME_NAMES = {
        2: "UPTREND",
        1: "UPTREND_TO_RANGE", 
        0: "RANGE",
        -1: "RANGE_TO_DOWNTREND",
        -2: "DOWNTREND"
    }
    
    REGIME_COLORS = {
        2: "#2a78d6",    # Blue
        1: "#9ec5f4",    # Light blue
        0: "#c3c2b7",    # Grey
        -1: "#f4a9a8",   # Light red
        -2: "#e34948"    # Red
    }
    
    REGIME_3CLASS = {
        1: "UPTREND",
        0: "RANGE",
        -1: "DOWNTREND"
    }
    
    def __init__(self, model_path=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to reversion_bundle.joblib
        """
        if model_path is None:
            # Default path relative to this file
            model_path = Path(__file__).parent.parent / "models" / "reversion_bundle.joblib"
        
        self.model_path = Path(model_path)
        self.bundle = None
        self.config = None
        
    def load_model(self):
        """Load the trained model bundle."""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False
        
        try:
            self.bundle = joblib.load(self.model_path)
            self.config = self.bundle['config']
            logger.info(f"Model loaded from {self.model_path}")
            logger.info(f"Trained through: {self.config['trained_through']}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def calculate_features(self, df_ohlc):
        """
        Calculate all features from OHLC data.
        Exact replication of make_frame() from training notebook.
        
        Args:
            df_ohlc: DataFrame with open, high, low, close, tickvol, spread
        
        Returns:
            DataFrame with features added
        """
        df = self._add_break_flags(df_ohlc.copy())
        
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        n = len(df)
        
        # True Range and ATR
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        w14 = dict(alpha=1 / 14, adjust=False, min_periods=14)
        atr = tr.ewm(**w14).mean()
        df['atr'] = atr
        
        # Directional Movement
        up, dn = h.diff(), -l.diff()
        pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
        mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
        pdi = 100 * pdm.ewm(**w14).mean() / atr
        mdi = 100 * mdm.ewm(**w14).mean() / atr
        adx = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(**w14).mean()
        di_diff = (pdi - mdi) / (pdi + mdi)
        
        df['adx'] = adx
        df['di_diff'] = di_diff
        
        # Efficiency Ratios
        W = self.config['W']
        df['er4'] = self._efficiency_ratio(c, 4)
        df['er8'] = self._efficiency_ratio(c, 8)
        df['er16'] = self._efficiency_ratio(c, W)
        
        # Regression slopes
        df['slope8'] = self._roll_slope(c, 8) * 7 / atr
        df['slope16'] = self._roll_slope(c, W) * (W - 1) / atr
        df['r2_16'] = self._roll_r2(c, W)
        
        # Range metrics
        rng_hi, rng_lo = h.rolling(W).max(), l.rolling(W).min()
        df['range_atr'] = (rng_hi - rng_lo) / atr
        df['pos'] = (c - rng_lo) / (rng_hi - rng_lo)
        
        # Overlap
        ov_num = (np.minimum(h, h.shift()) - np.maximum(l, l.shift())).clip(lower=0)
        ov_den = np.maximum(h, h.shift()) - np.minimum(l, l.shift())
        df['overlap'] = (ov_num / ov_den).rolling(8).mean()
        
        # Bollinger Band width
        df['bb_w'] = 4 * c.rolling(20).std() / atr
        
        # ATR ratio
        df['atr_ratio'] = atr / atr.rolling(96, min_periods=32).mean()
        
        # Volume ratio
        df['vol_ratio'] = df["tickvol"] / df["tickvol"].rolling(32, min_periods=16).mean()
        
        # Returns
        df['ret1'] = (c - c.shift(1)) / atr
        df['ret4'] = (c - c.shift(4)) / atr
        df['ret8'] = (c - c.shift(8)) / atr
        df['ret16'] = (c - c.shift(16)) / atr
        
        # Swing structure (confirmed)
        PIVOT_N = self.config['PIVOT_N']
        Hh, Ll = h.values, l.values
        ph, pl = [], []
        struct = np.zeros(n)
        
        for t in range(n):
            i = t - PIVOT_N
            if i >= PIVOT_N:
                if Hh[i] == Hh[i - PIVOT_N:t + 1].max():
                    ph.append(Hh[i])
                if Ll[i] == Ll[i - PIVOT_N:t + 1].min():
                    pl.append(Ll[i])
            if len(ph) >= 2 and len(pl) >= 2:
                struct[t] = ((1 if ph[-1] > ph[-2] else -1) + (1 if pl[-1] > pl[-2] else -1)) / 2
        
        df['struct'] = pd.Series(struct, index=df.index)
        
        # Trend score
        sgn = np.sign(c - c.shift(W))
        s_er = (sgn * df['er16'] / 0.35).clip(-1, 1)
        s_sl = (df['slope16'] / 3.0).clip(-1, 1)
        s_di = (di_diff / 0.35).clip(-1, 1)
        score = 0.30 * s_er + 0.30 * s_sl + 0.20 * s_di + 0.20 * df['struct']
        df['score'] = score
        
        # Score delta
        df['ds4'] = score - score.shift(4)
        
        # Regime state (5-class FSM)
        state = pd.Series(self._fsm(score.values, df["bar_valid"].values), index=df.index)
        df['state'] = state
        df['state_name'] = state.map(self.REGIME_NAMES)
        
        # 3-class state
        df['state3'] = pd.Series(
            np.where(state >= 2, 1, np.where(state <= -2, -1, 0)),
            index=df.index
        ).where(state.notna())
        
        # Trend momentum
        df['trend_mom'] = score.abs() - score.abs().shift(4)
        
        # Time features
        if self.config.get('USE_TIME_FEATURES', True):
            hr = df.index.hour + df.index.minute / 60
            df['hour_sin'] = np.sin(2 * np.pi * hr / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hr / 24)
            df['dow'] = df.index.dayofweek
        
        # Replace inf with nan
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Feature validity
        feats = self.config['FEATS']
        df['feat_valid'] = df[feats].notna().all(axis=1) & df['bar_valid']
        
        return df
    
    def _add_break_flags(self, df):
        """Add session break flags."""
        BREAK_HOURS = self.config['BREAK_HOURS']
        WARM = self.config['WARM']
        MASK_AFTER_BREAK = self.config['MASK_AFTER_BREAK']
        
        gap = df.index.to_series().diff()
        is_break = (gap > pd.Timedelta(hours=BREAK_HOURS)) | gap.isna()
        df["break_id"] = is_break.cumsum().values
        df["bars_since_break"] = df.groupby("break_id").cumcount().values
        
        if MASK_AFTER_BREAK:
            df["bar_valid"] = (df["bars_since_break"] >= WARM)
        else:
            df["bar_valid"] = (np.arange(len(df)) >= WARM)
        
        return df
    
    def _efficiency_ratio(self, series, period):
        """Calculate efficiency ratio."""
        net = (series - series.shift(period)).abs()
        path = series.diff().abs().rolling(period).sum()
        return net / path
    
    def _roll_slope(self, y, w):
        """Rolling linear regression slope."""
        x = np.arange(w) - (w - 1) / 2
        sxx = (x ** 2).sum()
        return y.rolling(w).apply(lambda a: np.dot(x, a) / sxx, raw=True)
    
    def _roll_r2(self, y, w):
        """Rolling R-squared."""
        x = np.arange(w) - (w - 1) / 2
        sxx = (x ** 2).sum()
        
        def f(a):
            b = np.dot(x, a) / sxx
            ss = ((a - a.mean()) ** 2).sum()
            return (b * b * sxx) / ss if ss > 0 else 0.0
        
        return y.rolling(w).apply(f, raw=True)
    
    def _fsm(self, score, valid):
        """
        Finite state machine for regime detection.
        Uses hysteresis to prevent flickering.
        """
        T_ENTER = self.config['T_ENTER']
        T_EXIT = self.config['T_EXIT']
        R_ENTER = self.config['R_ENTER']
        R_EXIT = self.config['R_EXIT']
        
        out = np.full(len(score), np.nan)
        cur = None
        
        for t in range(len(score)):
            s = score[t]
            if (not valid[t]) or np.isnan(s):
                cur = None
                continue
            
            a, g = abs(s), (1 if s > 0 else -1)
            raw = 2 * g if a >= T_ENTER else (0 if a < R_ENTER else g)
            
            if cur is None:
                cur = raw
            elif abs(cur) == 2:  # Trend: easier to stay than enter
                cur = cur if (a >= T_EXIT and g == np.sign(cur)) else raw
            elif cur == 0:  # Range: stay until clearly directional
                cur = 0 if a < R_EXIT else raw
            else:  # Transition zone
                cur = cur if (g == np.sign(cur) and R_ENTER <= a < T_ENTER) else raw
            
            out[t] = cur
        
        return out
    
    def predict_latest(self, df_ohlc):
        """
        Predict regime for the latest bar.
        
        Args:
            df_ohlc: DataFrame with OHLC data (at least 200+ recent bars)
        
        Returns:
            dict with predictions and probabilities
        """
        if self.bundle is None:
            if not self.load_model():
                return None
        
        # Calculate features
        fr = self.calculate_features(df_ohlc)
        
        # Get latest valid row
        valid_rows = fr[fr['feat_valid']]
        if valid_rows.empty:
            logger.warning("No valid rows for prediction (not enough bars?)")
            return None
        
        row = valid_rows.iloc[[-1]]
        
        # Extract features
        X = row[self.config['FEATS']].values
        
        # Predict regime in 1 hour (3-class)
        state_model = self.bundle['state_model']
        ps = state_model.predict_proba(X)[0]
        state_pred = state_model.predict(X)[0]
        
        # Predict direction (up before down)
        dir_model = self.bundle['dir_model']
        pdir = dir_model.predict_proba(X)[0]
        dir_classes = list(dir_model.classes_)
        
        # Build result
        result = {
            'bar_time': row.index[0],
            'close': float(row['close'].iloc[0]),
            'atr': float(row['atr'].iloc[0]),
            
            # Current regime
            'regime_now': row['state_name'].iloc[0],
            'regime_now_code': int(row['state'].iloc[0]),
            'regime_now_3class': self.REGIME_3CLASS.get(int(row['state3'].iloc[0]), 'UNKNOWN'),
            
            # Regime metrics
            'score': float(row['score'].iloc[0]),
            'trend_mom': float(row['trend_mom'].iloc[0]),
            
            # Predicted regime in 1 hour
            'regime_in_1h_pred': self.REGIME_3CLASS.get(int(state_pred), 'UNKNOWN'),
            'regime_in_1h_probs': {
                'DOWNTREND': float(ps[list(state_model.classes_).index(-1)]),
                'RANGE': float(ps[list(state_model.classes_).index(0)]),
                'UPTREND': float(ps[list(state_model.classes_).index(1)])
            },
            
            # Direction prediction
            'p_up_before_down': float(pdir[dir_classes.index(1)] if 1 in dir_classes else 0.5),
            'p_down_before_up': float(pdir[dir_classes.index(-1)] if -1 in dir_classes else 0.5),
            
            # Key features
            'features': {
                'er16': float(row['er16'].iloc[0]),
                'slope16': float(row['slope16'].iloc[0]),
                'adx': float(row['adx'].iloc[0]),
                'di_diff': float(row['di_diff'].iloc[0]),
                'range_atr': float(row['range_atr'].iloc[0]),
                'vol_ratio': float(row['vol_ratio'].iloc[0]),
                'atr_ratio': float(row['atr_ratio'].iloc[0])
            }
        }
        
        return result
    
    def predict_history(self, df_ohlc):
        """
        Predict regime for all bars in history.
        
        Returns:
            DataFrame with predictions
        """
        if self.bundle is None:
            if not self.load_model():
                return None
        
        # Calculate features
        fr = self.calculate_features(df_ohlc)
        
        # Filter valid rows
        valid_mask = fr['feat_valid']
        if valid_mask.sum() == 0:
            logger.warning("No valid rows for prediction")
            return None
        
        X = fr.loc[valid_mask, self.config['FEATS']].values
        
        # Predict regime in 1 hour
        state_model = self.bundle['state_model']
        regime_pred = state_model.predict(X)
        regime_proba = state_model.predict_proba(X)
        
        # Predict direction
        dir_model = self.bundle['dir_model']
        dir_proba = dir_model.predict_proba(X)
        dir_classes = list(dir_model.classes_)
        
        # Create result dataframe
        result = fr.loc[valid_mask].copy()
        result['regime_pred_1h'] = regime_pred
        result['regime_pred_1h_name'] = [self.REGIME_3CLASS.get(int(x), 'UNKNOWN') for x in regime_pred]
        
        # Add 3-class current regime name for display
        result['regime_now_3class'] = result['state3'].map(lambda x: self.REGIME_3CLASS.get(int(x), 'UNKNOWN') if pd.notna(x) else 'UNKNOWN')
        
        # Add probabilities
        state_classes = list(state_model.classes_)
        result['prob_down_1h'] = regime_proba[:, state_classes.index(-1)]
        result['prob_range_1h'] = regime_proba[:, state_classes.index(0)]
        result['prob_up_1h'] = regime_proba[:, state_classes.index(1)]
        
        if 1 in dir_classes and -1 in dir_classes:
            result['prob_up_before_down'] = dir_proba[:, dir_classes.index(1)]
            result['prob_down_before_up'] = dir_proba[:, dir_classes.index(-1)]
        
        return result


def test_predictor():
    """Test the predictor with sample data."""
    import sys
    sys.path.append(str(Path(__file__).parent))
    from mt5_data_loader import MT5DataLoader
    
    print("Testing Regime Predictor...")
    print("=" * 60)
    
    # Load model
    predictor = RegimePredictor()
    if not predictor.load_model():
        print("❌ Failed to load model")
        return
    
    print("✓ Model loaded successfully")
    print(f"  Model type: {predictor.config['model']}")
    print(f"  Features: {len(predictor.config['FEATS'])}")
    
    # Get MT5 data
    print("\nFetching MT5 data...")
    loader = MT5DataLoader(symbol='XAUUSDm', timeframe=mt5.TIMEFRAME_M15)
    if not loader.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    df = loader.get_rates(n_bars=500)
    loader.shutdown()
    
    if df is None:
        print("❌ Failed to get data")
        return
    
    print(f"✓ Fetched {len(df)} bars")
    
    # Predict latest
    print("\nPredicting current regime...")
    result = predictor.predict_latest(df)
    
    if result is None:
        print("❌ Prediction failed")
        return
    
    print("\n" + "=" * 60)
    print("CURRENT MARKET REGIME")
    print("=" * 60)
    print(f"Time: {result['bar_time']}")
    print(f"Close: {result['close']:.2f}")
    print(f"ATR: {result['atr']:.2f}")
    print(f"\nCurrent Regime: {result['regime_now']} ({result['regime_now_3class']})")
    print(f"Trend Score: {result['score']:.3f}")
    print(f"Trend Momentum: {result['trend_mom']:.3f}")
    
    print(f"\nPredicted Regime in 1 Hour: {result['regime_in_1h_pred']}")
    print("Probabilities:")
    for regime, prob in result['regime_in_1h_probs'].items():
        bar = "█" * int(prob * 50)
        print(f"  {regime:12s}: {prob:5.1%} {bar}")
    
    print(f"\nDirection Probabilities:")
    print(f"  Up before Down: {result['p_up_before_down']:.1%}")
    print(f"  Down before Up: {result['p_down_before_up']:.1%}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    import MetaTrader5 as mt5
    logging.basicConfig(level=logging.INFO)
    test_predictor()
