# Market Regime Monitor - Quick Start Guide 🚀

## ⚡ 5-Minute Setup

### Step 1: Verify Setup ✓
```bash
cd c:\Users\MYCkey98\Documents\Reversion\regime_monitor
python test_setup.py
```

**Expected output**: ✅ Setup complete!

### Step 2: Launch Dashboard 🚀

**Option A - Command Line:**
```bash
streamlit run streamlit_app.py
```

**Option B - Windows:**
Double-click `launch_dashboard.bat`

### Step 3: Use Dashboard 📊

1. **Configure** (left sidebar):
   - Symbol: Select your symbol (XAUUSD, EURUSD, etc.)
   - Timeframe: M15 recommended
   - Bars: 500 (default is good)

2. **Load Data**:
   - Click "🔄 Fetch Data" button
   - Wait 5-10 seconds for analysis

3. **View Results**:
   - Current regime status (top cards)
   - 1-hour prediction with probabilities
   - Interactive charts
   - Statistics and analytics

## 📱 Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│  ⚙️ Sidebar                  📊 Market Regime Monitor    │
│                                                          │
│  Symbol: XAUUSD              [Current] [Predicted]      │
│  Timeframe: M15              UPTREND → RANGE (60%)      │
│  Bars: 500                                              │
│                              Score: 0.823               │
│  Display Settings            Price: 2745.25             │
│  Chart bars: 200             ATR: 2.34                  │
│                                                         │
│  🔄 Auto Refresh            ──────────────────────      │
│  ☐ Enable                                               │
│                             🔮 1-Hour Prediction         │
│                             [Gauges: Down|Range|Up]     │
│                                                         │
│                             📈 Direction                 │
│                             Up: 55% | Down: 45%         │
│                                                         │
│                             ──────────────────────      │
│                                                         │
│                             📊 Price & Regime Chart     │
│                             [Interactive Candlestick]   │
│                             [Trend Score Line]          │
│                                                         │
│                             ──────────────────────      │
│                                                         │
│                             📊 Distribution | Features   │
│                             [Bar Chart]    | [Values]   │
│                                                         │
│                             📈 Regime Timeline          │
│                             [Area Chart]                │
│                                                         │
│                             📊 Statistics               │
│                             [Duration|Strength|etc]     │
│                                                         │
│                             📋 Recent Data Table        │
│                             [Expandable]                │
└─────────────────────────────────────────────────────────┘
```

## 🎯 Reading the Dashboard

### Status Cards (Top)

**Current Regime** (colored box):
- 🔵 Blue = UPTREND (strong up)
- 🔵 Light Blue = UPTREND_TO_RANGE (weakening)
- ⚪ Grey = RANGE (sideways)
- 🔴 Light Red = RANGE_TO_DOWNTREND (weakening)
- 🔴 Red = DOWNTREND (strong down)

**Predicted Regime** (1 hour ahead):
- Shows simplified 3-class prediction
- High confidence if probability > 70%

**Trend Score**:
- Positive = uptrend
- Negative = downtrend
- Near zero = range
- Delta shows if strengthening (+) or fading (-)

### Prediction Gauges

Three circular gauges showing probabilities:
- **Green zone** (>50%): High confidence
- **Yellow zone** (33-50%): Moderate
- **Red zone** (<33%): Low confidence

### Main Chart

- **Candlesticks**: Price action
- **Background colors**: Current regime at each bar
- **Bottom panel**: Trend score with thresholds
- **Hover**: Shows details for each bar

### Distribution Chart

Shows how often each regime occurs:
- Balanced = healthy variety
- Dominated by one = trending/ranging market

### Feature Values

Key indicators that drive predictions:
- **er16**: Trend efficiency (higher = cleaner trend)
- **slope16**: Trend slope (+ up, - down)
- **adx**: Trend strength (>25 = strong)
- **di_diff**: Directional bias
- **range_atr**: Range size (lower = tighter)
- **vol_ratio**: Volume compared to average
- **atr_ratio**: Volatility vs average

## 🎓 Common Questions

### Q: What symbol can I use?
**A**: Any symbol in your MT5 terminal. The model was trained on XAUUSD M15, but works on similar markets. Test before live trading on other symbols.

### Q: What timeframe is best?
**A**: M15 (model was trained on this). M5 and M30 work okay. H1+ may need model retraining.

### Q: How often should I refresh?
**A**: 
- Manual: Every 15-60 minutes
- Auto-refresh: 60 seconds (not recommended for demo)
- Live trading: 30-60 seconds

### Q: What's a "good" prediction?
**A**:
- ✅ One regime > 70% probability
- ✅ Trend score aligned with prediction
- ✅ Multiple bars in same regime
- ⚠️ All close to 33% = uncertain

### Q: How do I use this for trading?
**A**: See integration example below.

## 🤝 Trading Integration

### Example: Align Strategy with Regime

```python
# Pseudo-code for integration
current_regime = get_current_regime()
predicted_regime = get_predicted_regime()
confidence = max(prediction_probabilities)

# Strategy logic
if current_regime == "UPTREND" and confidence > 0.7:
    # Strong uptrend: look for long entries
    if pullback_to_support and rsi < 50:
        enter_long()
        
elif current_regime == "DOWNTREND" and confidence > 0.7:
    # Strong downtrend: look for short entries
    if rally_to_resistance and rsi > 50:
        enter_short()
        
elif current_regime == "RANGE":
    # Range: mean reversion or stay out
    if price_at_range_extremes:
        fade_the_move()
    else:
        wait_for_breakout()
        
else:
    # Transition or uncertain: reduce size or wait
    reduce_position_size()
```

### Position Sizing by Regime

| Regime | Confidence | Position Size |
|--------|------------|---------------|
| UPTREND | > 70% | 100% (full) |
| UPTREND | 50-70% | 50% (half) |
| RANGE | > 60% | 50% (mean-revert only) |
| Transition | Any | 25% (very small) |
| Uncertain | < 50% | 0% (no trade) |

### Exit Rules

- **Regime change**: Exit when regime flips (e.g., UP → RANGE)
- **Fading trend**: Exit when trend_mom < -0.1
- **Low confidence**: Exit when next prediction shows < 40% for current direction
- **Duration**: Consider exit after 20+ bars in same regime

## 🔧 Troubleshooting

### Dashboard won't start
```bash
# Check if port 8501 is busy
netstat -ano | findstr :8501

# Use different port
streamlit run streamlit_app.py --server.port 8502
```

### "Failed to initialize MT5"
1. Open MetaTrader 5
2. Login to account
3. Add symbol to Market Watch
4. Try again

### "Model not found"
```bash
# Check if model exists
dir ..\models\reversion_bundle.joblib

# If missing, run training notebook:
# notebooks/training/Reversion_regime_model.ipynb
```

### "No valid rows for prediction"
- Need at least 200 bars for feature calculation
- Increase `n_bars` to 500+
- Check if symbol has sufficient history

### Dashboard is slow
- Reduce historical bars (300 instead of 500)
- Reduce chart bars (100 instead of 200)
- Disable auto-refresh when not needed
- Close other Streamlit sessions

## 📊 Example Session

```
1. Launch: streamlit run streamlit_app.py
   → Browser opens to http://localhost:8501

2. Configure:
   - Symbol: XAUUSD
   - Timeframe: M15
   - Bars: 500

3. Click "Fetch Data"
   → Loading... (5-10 seconds)
   
4. Review Results:
   ✓ Current: DOWNTREND (Red)
   ✓ Predicted: DOWNTREND (75%)
   ✓ Score: -0.82 (strong down)
   ✓ Trend Mom: -0.14 (fading slightly)
   
5. Interpret:
   - Strong downtrend confirmed
   - High confidence (75%)
   - Look for short entries on rallies
   - Watch for trend_mom turning positive

6. Trade Decision:
   → Wait for pullback to EMA15
   → Enter short if RSI > 50
   → SL: 1.5 × ATR above entry
   → TP: 5.0 × ATR below entry
```

## 🎯 Next Steps

1. ✅ **Monitor for a day** - Get familiar with the interface
2. ✅ **Compare with price** - See how regimes align with market
3. ✅ **Note transitions** - When does regime change?
4. ✅ **Check confidence** - Are high-probability predictions accurate?
5. ✅ **Paper trade** - Test integration with strategy
6. ✅ **Refine rules** - Adjust based on observations

## 📚 Further Reading

- **Full Documentation**: `README.md`
- **Technical Details**: `PROJECT_OVERVIEW.md`
- **Model Training**: `../notebooks/training/Reversion_regime_model.ipynb`
- **Theory**: `../documentation/theory/01_market_regime_theory.md`

## 🚀 Ready to Start!

```bash
cd c:\Users\MYCkey98\Documents\Reversion\regime_monitor
streamlit run streamlit_app.py
```

**Dashboard URL**: http://localhost:8501

---

**Happy Trading! 📈💰**
