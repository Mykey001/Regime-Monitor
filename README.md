# Market Regime Monitor 📊

Real-time market regime detection and visualization using trained machine learning model.

## Features

- ✅ **Real-time regime detection** - Predicts current and future market regimes
- 📈 **Interactive charts** - Visualize price action with regime coloring
- 🎯 **Prediction probabilities** - See confidence levels for regime predictions
- 📊 **Comprehensive analytics** - Regime statistics, transitions, and feature importance
- 🔄 **Auto-refresh** - Optional automatic data updates
- 🎨 **Beautiful GUI** - Built with Streamlit and Plotly

## Market Regimes

The model detects 5 distinct market regimes:

1. **UPTREND** (Blue) - Strong upward momentum
2. **UPTREND_TO_RANGE** (Light Blue) - Transitioning from uptrend
3. **RANGE** (Grey) - Sideways/consolidation
4. **RANGE_TO_DOWNTREND** (Light Red) - Transitioning to downtrend
5. **DOWNTREND** (Red) - Strong downward momentum

## Installation

### 1. Install Dependencies

```bash
cd regime_monitor
pip install -r requirements.txt
```

### 2. Ensure MT5 is Running

Make sure MetaTrader 5 is installed and running with the symbol you want to analyze.

### 3. Verify Model is Available

The model should be at: `../models/reversion_bundle.joblib`

## Usage

### Start the Dashboard

```bash
streamlit run streamlit_app.py
```

This will open the dashboard in your default browser (usually `http://localhost:8501`).

### Using the Dashboard

1. **Configure Settings** (left sidebar):
   - Select symbol (XAUUSD, EURUSD, etc.)
   - Choose timeframe (M1, M5, M15, M30, H1, H4)
   - Set number of historical bars to load
   - Adjust chart display settings

2. **Fetch Data**:
   - Click "🔄 Fetch Data" button to load data from MT5
   - Wait for analysis to complete

3. **View Results**:
   - Current regime status
   - 1-hour regime prediction with probabilities
   - Interactive price chart with regime coloring
   - Regime statistics and transitions
   - Key feature values

4. **Auto-Refresh** (optional):
   - Enable in sidebar to automatically update data
   - Set refresh interval (10-300 seconds)

## Testing Individual Components

### Test MT5 Connection

```bash
python mt5_data_loader.py
```

Expected output:
```
Testing MT5 Connection...
==================================================
✓ MT5 initialized
✓ Symbol Info: ...
✓ Current Tick: ...
✓ Historical Data: ...
```

### Test Regime Predictor

```bash
python regime_predictor.py
```

Expected output:
```
Testing Regime Predictor...
============================================================
✓ Model loaded successfully
  Model type: rf
  Features: 25

Fetching MT5 data...
✓ Fetched 500 bars

Predicting current regime...
============================================================
CURRENT MARKET REGIME
============================================================
Time: 2026-10-30 20:45:00
Close: 2745.25
ATR: 2.34

Current Regime: DOWNTREND (DOWNTREND)
Trend Score: -0.823
Trend Momentum: -0.145

Predicted Regime in 1 Hour: DOWNTREND
Probabilities:
  DOWNTREND   : 75.2% ████████████████████████████████████████
  RANGE       : 24.4% ████████████
  UPTREND     :  0.4%
```

## Architecture

### Components

1. **mt5_data_loader.py** - Fetches real-time data from MetaTrader 5
2. **regime_predictor.py** - Loads model and generates predictions
3. **streamlit_app.py** - Main dashboard application

### Data Flow

```
MT5 Terminal
    ↓
MT5DataLoader (fetch OHLCV data)
    ↓
RegimePredictor (calculate features)
    ↓
Trained Model (predict regime)
    ↓
Streamlit GUI (visualize)
```

## Model Information

- **Training Data**: XAUUSD M15, 2024-2026 (67k bars)
- **Model Type**: Random Forest (can be changed to LightGBM or HGB)
- **Accuracy**: 76.6% for 1-hour regime prediction
- **Features**: 25+ technical indicators (ER, slope, ADX, structure, etc.)

### Key Features Used

- Efficiency ratios (ER4, ER8, ER16)
- Regression slopes (slope8, slope16, R²)
- Directional movement (ADX, DI difference)
- Range metrics (range_atr, position)
- Momentum (returns 1/4/8/16 bars)
- Structure (swing point analysis)
- Volatility (ATR ratio, BB width)
- Volume ratio
- Time features (hour sin/cos, day of week)

## Interpretation Guide

### Trend Score
- **> 0.55**: Strong uptrend
- **0.18 to 0.55**: Weak uptrend / transitioning
- **-0.18 to 0.18**: Range-bound
- **-0.55 to -0.18**: Weak downtrend / transitioning
- **< -0.55**: Strong downtrend

### Trend Momentum
- **Positive**: Trend strengthening
- **Negative**: Trend weakening/fading

### Prediction Confidence
- **> 70%**: High confidence
- **50-70%**: Moderate confidence
- **< 50%**: Low confidence (uncertain regime)

## Troubleshooting

### "Failed to initialize MT5"
- Ensure MetaTrader 5 is running
- Check that the symbol is available in Market Watch
- Verify MT5 terminal is not restricted

### "Failed to load model"
- Check model path: `../models/reversion_bundle.joblib`
- Verify the model file exists
- Re-run training notebook if needed

### "No valid rows for prediction"
- Need at least 200+ bars for feature calculation
- Increase `n_bars` parameter
- Check data quality (no gaps/missing values)

### Slow Performance
- Reduce number of historical bars
- Reduce chart display bars
- Increase auto-refresh interval

## Directory Structure

```
regime_monitor/
├── streamlit_app.py        # Main dashboard
├── regime_predictor.py     # Prediction engine
├── mt5_data_loader.py      # MT5 data interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Cached data (auto-created)
├── logs/                  # Log files (auto-created)
└── cache/                 # Streamlit cache (auto-created)
```

## Future Enhancements

- [ ] Export predictions to CSV
- [ ] Alert notifications when regime changes
- [ ] Multiple symbol monitoring
- [ ] Historical performance tracking
- [ ] Integration with trading strategy
- [ ] Backtesting interface

## Support

For issues or questions:
1. Check logs in `logs/regime_monitor.log`
2. Verify MT5 connection with test scripts
3. Ensure model is properly trained

## License

Part of the Reversion trading system project.
# Regime-Monitor
