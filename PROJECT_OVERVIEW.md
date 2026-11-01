# Market Regime Monitor - Project Overview

## 🎯 Project Purpose

A real-time market regime detection and visualization system that uses a trained machine learning model to:
1. Identify the current market regime (UPTREND, RANGE, DOWNTREND, or transitions)
2. Predict the regime 1 hour ahead with probability estimates
3. Provide comprehensive market analytics and KPIs
4. Visualize regime changes on interactive charts

## 📊 Model Performance

### Training Results (from notebook)
- **Dataset**: XAUUSD M15, 67,031 bars (2024-2026)
- **Validation**: 8-fold walk-forward (purged, non-overlapping)
- **Model**: Random Forest (can switch to LightGBM or HGBoost)

### Task B3: Regime Prediction (1 hour ahead)
| Model | Accuracy | Balanced Acc | Macro F1 |
|-------|----------|--------------|----------|
| **LightGBM** | **76.6%** | **76.1%** | **76.8%** |
| HGBoost | 76.4% | 75.8% | 76.6% |
| Random Forest | 74.8% | 77.0% | 75.4% |
| Persistence | 70.0% | 70.4% | 70.4% |

✅ **Beats baseline by 6.6 percentage points**
✅ **Consistent across years** (2024: 76.4%, 2025: 76.6%, 2026: 76.7%)

### Task C: Direction Prediction (Triple Barrier)
| Model | Threshold | Trades | Hit Rate | Avg P&L | Total P&L |
|-------|-----------|--------|----------|---------|-----------|
| **LightGBM** | 0.60 | 1,230 | **53.0%** | **$0.38** | **+$470** |
| Random Forest | 0.55 | 719 | 52.4% | $0.40 | +$287 |
| Random Forest | 0.60 | 56 | 53.6% | $0.81 | +$45 |

✅ **Profitable after spread costs**
✅ **Statistically significant edge** (p < 0.01)

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MetaTrader 5 Terminal                     │
│                   (Real-time Market Data)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                    MT5DataLoader                             │
│  • Connects to MT5 API                                       │
│  • Fetches OHLCV data (configurable bars)                   │
│  • Handles multiple symbols & timeframes                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  RegimePredictor                             │
│  • Loads trained model bundle                                │
│  • Calculates 25+ technical features                         │
│  • Implements FSM for regime detection                       │
│  • Generates predictions & probabilities                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                 Streamlit Dashboard                          │
│  • Interactive web interface                                 │
│  • Real-time charts with Plotly                             │
│  • KPI cards & analytics                                     │
│  • Auto-refresh capability                                   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
regime_monitor/
│
├── streamlit_app.py              # Main dashboard (Streamlit GUI)
├── regime_predictor.py           # Prediction engine & feature calculation
├── mt5_data_loader.py            # MT5 data interface
│
├── requirements.txt              # Python dependencies
├── README.md                     # User guide
├── PROJECT_OVERVIEW.md           # This file
│
├── test_setup.py                 # Setup verification script
├── launch_dashboard.bat          # Windows launcher
│
├── data/                         # Cached data (auto-created)
├── logs/                         # Application logs
│   └── regime_monitor.log
└── cache/                        # Streamlit cache
```

## 🔬 Feature Engineering

The system calculates 25+ features from raw OHLC data:

### 1. Efficiency Ratios (3 features)
- ER4, ER8, ER16: Net move ÷ total path distance
- Measures trend efficiency vs noise

### 2. Regression Metrics (3 features)
- slope8, slope16: Linear regression slopes (normalized by ATR)
- r2_16: R-squared of 16-bar regression

### 3. Directional Movement (2 features)
- ADX: Average Directional Index (trend strength)
- di_diff: Difference between +DI and -DI (trend direction)

### 4. Range & Position (3 features)
- range_atr: 16-bar range in ATR units
- pos: Position within range (0 = bottom, 1 = top)
- overlap: Bar-to-bar overlap ratio

### 5. Volatility (3 features)
- atr_ratio: Current ATR vs rolling average
- bb_w: Bollinger Band width (normalized)
- vol_ratio: Volume vs rolling average

### 6. Momentum (4 features)
- ret1, ret4, ret8, ret16: Returns over N bars (in ATR units)

### 7. Structure (1 feature)
- struct: Swing point analysis (higher highs & lows)

### 8. Composite Metrics (2 features)
- score: Weighted combination of ER, slope, DI, structure
- ds4: 4-bar change in trend score

### 9. Regime State (1 feature)
- state: FSM output (5 classes with hysteresis)

### 10. Time Features (3 features, optional)
- hour_sin, hour_cos: Circular encoding of hour
- dow: Day of week

## 🎨 Regime Definitions

### Five-State Model (with hysteresis)

| State | Code | Color | Description | Entry | Exit |
|-------|------|-------|-------------|-------|------|
| **UPTREND** | +2 | Blue | Strong uptrend | score ≥ 0.55 | score < 0.40 |
| **UPTREND_TO_RANGE** | +1 | Light Blue | Weakening up | 0.18 ≤ score < 0.55 | - |
| **RANGE** | 0 | Grey | Sideways | \|score\| < 0.18 | \|score\| ≥ 0.28 |
| **RANGE_TO_DOWNTREND** | -1 | Light Red | Weakening | -0.55 < score ≤ -0.18 | - |
| **DOWNTREND** | -2 | Red | Strong downtrend | score ≤ -0.55 | score > -0.40 |

**Hysteresis**: Harder to enter a regime than to stay in it (prevents flickering)

### Three-State Simplified

| State | Mapping | Use Case |
|-------|---------|----------|
| **UPTREND** | +2, +1 | Trade long signals |
| **RANGE** | 0 | Avoid trading or mean-revert |
| **DOWNTREND** | -2, -1 | Trade short signals |

## 📈 Dashboard Features

### 1. Current Status Cards
- Current regime (5-class)
- Predicted regime in 1 hour
- Trend score with momentum delta
- Current price & ATR

### 2. Prediction Gauges
- Three probability gauges (Down, Range, Up)
- Color-coded confidence levels
- Direction probabilities (up before down / down before up)

### 3. Interactive Price Chart
- Candlestick chart
- Regime background coloring
- Trend score subplot with thresholds
- Configurable time window

### 4. Analytics Panels
- Regime distribution histogram
- Key feature values bar chart
- Regime timeline (last N bars)
- Transition probability matrix

### 5. Statistics
- Current regime duration
- Average trend strength
- Average volatility
- Recent transitions

### 6. Data Table
- Last 50 bars with key columns
- Formatted metrics
- Exportable

## 🔧 Configuration Options

### Symbol Selection
- XAUUSD (Gold)
- EURUSD, GBPUSD, USDJPY (Forex)
- Any MT5 symbol (if data compatible)

### Timeframe Options
- M1, M5, M15 (recommended), M30, H1, H4
- Model trained on M15 (best results)
- Other timeframes work but may need retraining

### Display Settings
- Historical bars: 100-1000 (affects loading time)
- Chart bars: 50-500 (affects visual clarity)
- Auto-refresh: 10-300 seconds

## 🚀 Usage Workflow

### Setup (One-time)
1. Install dependencies: `pip install -r requirements.txt`
2. Verify model exists: `../models/reversion_bundle.joblib`
3. Run test: `python test_setup.py`

### Daily Use
1. Start MT5 terminal
2. Launch dashboard: `streamlit run streamlit_app.py`
3. Configure symbol & timeframe
4. Click "Fetch Data"
5. Monitor regime & predictions
6. Optional: Enable auto-refresh

### Integration with Trading
1. Check current regime before trading
2. Align strategy with regime:
   - **UPTREND**: Look for pullback longs
   - **DOWNTREND**: Look for rally shorts
   - **RANGE**: Mean-reversion or avoid
3. Use predicted regime for position sizing
4. Monitor transitions for exits

## 📊 Interpretation Guide

### High Confidence Signals
✅ Regime probability > 70%
✅ Trend score aligned with regime
✅ Multiple consecutive bars in same regime
✅ Features showing clear structure (high ER, aligned slope)

### Low Confidence / Uncertain
⚠️ Probabilities close to 33% (uniform)
⚠️ Trend score near thresholds (-0.2 to 0.2)
⚠️ Recent regime flip (< 4 bars)
⚠️ High volatility (atr_ratio > 1.5)

### Trading Implications

| Scenario | Action |
|----------|--------|
| Current = UPTREND, Predicted = UPTREND (>70%) | Strong long bias |
| Current = UPTREND, Predicted = RANGE (>60%) | Take profits on longs |
| Current = RANGE, Predicted = UPTREND (>60%) | Prepare long entries |
| Current regime duration > 20 bars | Mature, watch for reversal |
| Trend momentum < 0 (fading) | Caution, trend weakening |

## 🔍 Model Insights

### Most Important Features (Permutation Importance)
1. **score** - Composite trend score (most predictive)
2. **state** - Current regime (strong persistence)
3. **er16** - 16-bar efficiency ratio
4. **slope16** - 16-bar regression slope
5. **ds4** - Trend score momentum
6. **struct** - Swing structure
7. **adx** - Trend strength
8. **di_diff** - Directional bias

### Why It Works
- **Regime persistence**: Markets stay in regimes longer than random
- **Multi-timeframe**: Combines short (4-bar) and medium (16-bar) signals
- **Hysteresis**: Prevents false signals from noise
- **Structure**: Uses confirmed swing points (not lagging)
- **Walk-forward validated**: No lookahead bias

### Limitations
- **Lag**: Reacts ~8 bars after true regime change (deliberate tradeoff)
- **Whipsaws**: Can still flip in choppy markets (use confidence levels)
- **Market drift**: Performance may degrade over time (retrain periodically)
- **Symbol-specific**: Trained on XAUUSD (test before using on other symbols)

## 🛠️ Customization

### Change Model Type
In `regime_predictor.py`, the model is loaded from the bundle.
To retrain with a different model:
1. Open `Reversion_regime_model.ipynb`
2. Change `FINAL_MODEL_NAME = "lgbm"` (or "hgb")
3. Re-run section 9 to save new bundle

### Adjust Regime Thresholds
In the notebook config:
- `T_ENTER = 0.55` (harder to enter trends)
- `T_EXIT = 0.40` (easier to stay in trends)
- `R_ENTER = 0.18` (enter range states)
- `R_EXIT = 0.28` (exit range states)

### Add New Features
In `regime_predictor.py`, `calculate_features()`:
1. Compute new indicator
2. Add to feature list
3. Retrain model with new features

### Custom Visualizations
In `streamlit_app.py`:
- Modify `plot_regime_chart()` for different chart styles
- Add new plot functions
- Create custom KPI cards

## 📝 Development Notes

### Code Quality
- Type hints for key functions
- Logging for debugging
- Error handling for MT5 failures
- Caching for performance

### Performance
- Vectorized feature calculation (pandas/numpy)
- Efficient data structures
- Minimal recomputation
- Streamlit caching for plots

### Testing
- `test_setup.py`: Verifies installation
- `mt5_data_loader.py`: Can run standalone
- `regime_predictor.py`: Can run standalone

## 🎓 Learning Resources

### Understanding Regimes
- Read: `documentation/theory/01_market_regime_theory.md`
- Read: `documentation/methodology/00_PROJECT_SUMMARY.md`

### Feature Details
- Read: `documentation/theory/02_feature_engineering_specification.md`

### Model Training
- Notebook: `notebooks/training/Reversion_regime_model.ipynb`
- See sections 3-8 for detailed explanations

## 🚦 Status & Roadmap

### ✅ Completed
- [x] Model training & validation
- [x] Feature engineering pipeline
- [x] MT5 data integration
- [x] Streamlit dashboard
- [x] Real-time predictions
- [x] Interactive visualizations
- [x] Documentation

### 🔄 In Progress
- [ ] Alert system (notify on regime change)
- [ ] Export predictions to CSV
- [ ] Performance tracking over time

### 📋 Future Enhancements
- [ ] Multi-symbol monitoring
- [ ] Strategy backtesting interface
- [ ] Auto-trading integration
- [ ] Mobile-friendly UI
- [ ] Cloud deployment option
- [ ] Email/SMS alerts
- [ ] Custom indicator builder

## 🤝 Integration with Trading Strategy

This regime monitor is designed to work alongside the `mt5_optimized_strategy.py`:

```python
# Example integration
from regime_monitor.regime_predictor import RegimePredictor

predictor = RegimePredictor()
prediction = predictor.predict_latest(recent_bars)

if prediction['regime_now_3class'] == 'UPTREND':
    # Only take long signals
    if ema_cross_up and rsi < 50:
        open_long()
elif prediction['regime_now_3class'] == 'DOWNTREND':
    # Only take short signals
    if ema_cross_down and rsi > 50:
        open_short()
else:
    # Range: stricter filters or avoid
    pass
```

## 📞 Support

For issues:
1. Check `logs/regime_monitor.log`
2. Run `python test_setup.py`
3. Verify MT5 connection
4. Review README.md troubleshooting section

---

**Project**: Reversion Trading System
**Component**: Market Regime Monitor
**Version**: 1.0
**Last Updated**: 2026-10-31
