"""
Market Regime Monitor - Streamlit Dashboard
===========================================

Real-time market regime detection and visualization using trained ML model.
Displays current regime, predictions, and key performance indicators.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from regime_predictor import RegimePredictor
from mt5_data_loader import MT5DataLoader
import MetaTrader5 as mt5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/regime_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Market Regime Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Regime colors
REGIME_COLORS = {
    "UPTREND": "#2a78d6",
    "UPTREND_TO_RANGE": "#9ec5f4",
    "RANGE": "#c3c2b7",
    "RANGE_TO_DOWNTREND": "#f4a9a8",
    "DOWNTREND": "#e34948"
}

REGIME_3_COLORS = {
    "UPTREND": "#2a78d6",
    "RANGE": "#c3c2b7",
    "DOWNTREND": "#e34948"
}


def initialize_session_state():
    """Initialize session state variables."""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'loader' not in st.session_state:
        st.session_state.loader = None
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None


def load_predictor():
    """Load the regime predictor model."""
    if st.session_state.predictor is None:
        predictor = RegimePredictor()
        if predictor.load_model():
            st.session_state.predictor = predictor
            return True
        return False
    return True


def initialize_mt5(symbol, timeframe):
    """Initialize MT5 connection."""
    if st.session_state.loader is None:
        loader = MT5DataLoader(symbol=symbol, timeframe=timeframe)
        if loader.initialize():
            st.session_state.loader = loader
            return True
        return False
    return True


def get_market_data(n_bars=500):
    """Fetch market data from MT5."""
    if st.session_state.loader is None:
        return None
    
    df = st.session_state.loader.get_rates(n_bars=n_bars)
    return df


def plot_regime_chart(df_results, n_bars=200):
    """Create price chart with regime coloring."""
    # Get last N bars
    df = df_results.tail(n_bars).copy()
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price & Regime', 'Trend Score')
    )
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add regime background colors
    for regime_name, color in REGIME_COLORS.items():
        regime_data = df[df['state_name'] == regime_name]
        if not regime_data.empty:
            for idx in regime_data.index:
                fig.add_vrect(
                    x0=idx, x1=idx + pd.Timedelta(minutes=15),
                    fillcolor=color, opacity=0.2,
                    layer="below", line_width=0,
                    row=1, col=1
                )
    
    # Add trend score
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['score'],
            name='Trend Score',
            line=dict(color='#52514e', width=2),
            mode='lines'
        ),
        row=2, col=1
    )
    
    # Add threshold lines
    config = st.session_state.predictor.config
    fig.add_hline(y=config['T_ENTER'], line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    fig.add_hline(y=-config['T_ENTER'], line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    fig.add_hline(y=0, line_color="#c3c2b7", opacity=0.8, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title='Market Regime Detection',
        xaxis_rangeslider_visible=False,
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    
    return fig


def plot_regime_distribution(df_results):
    """Plot regime distribution over time."""
    # Count regimes
    regime_counts = df_results['state_name'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=regime_counts.index,
            y=regime_counts.values,
            marker_color=[REGIME_COLORS.get(x, '#999999') for x in regime_counts.index],
            text=regime_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Regime Distribution (All Data)',
        xaxis_title='Regime',
        yaxis_title='Bar Count',
        height=300
    )
    
    return fig


def plot_prediction_probabilities(prediction):
    """Plot prediction probabilities as gauge charts."""
    probs = prediction['regime_in_1h_probs']
    
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=['Downtrend', 'Range', 'Uptrend']
    )
    
    # Downtrend
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=probs['DOWNTREND'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Down"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': REGIME_3_COLORS['DOWNTREND']},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ),
        row=1, col=1
    )
    
    # Range
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=probs['RANGE'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Range"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': REGIME_3_COLORS['RANGE']},
                'threshold': {
                    'line': {'color': "gray", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ),
        row=1, col=2
    )
    
    # Uptrend
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=probs['UPTREND'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Up"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': REGIME_3_COLORS['UPTREND']},
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def plot_feature_importance(prediction):
    """Plot key feature values."""
    features = prediction['features']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(features.keys()),
        y=list(features.values()),
        marker_color='steelblue',
        text=[f"{v:.3f}" for v in features.values()],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Key Feature Values',
        xaxis_title='Feature',
        yaxis_title='Value',
        height=300
    )
    
    return fig


def plot_regime_timeline(df_results, n_bars=100):
    """Plot regime changes over time."""
    df = df_results.tail(n_bars).copy()
    
    # Create numerical regime for plotting
    regime_map = {'DOWNTREND': -2, 'RANGE_TO_DOWNTREND': -1, 'RANGE': 0, 
                  'UPTREND_TO_RANGE': 1, 'UPTREND': 2}
    df['regime_num'] = df['state_name'].map(regime_map)
    
    fig = go.Figure()
    
    # Plot regime as area
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['regime_num'],
        fill='tozeroy',
        name='Regime',
        line=dict(color='steelblue', width=2),
        fillcolor='rgba(70, 130, 180, 0.3)'
    ))
    
    # Add regime labels
    fig.update_layout(
        title=f'Regime Timeline (Last {n_bars} bars)',
        xaxis_title='Time',
        yaxis_title='Regime',
        height=300,
        yaxis=dict(
            tickmode='array',
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=['DOWNTREND', 'RANGE_TO_DOWN', 'RANGE', 'UP_TO_RANGE', 'UPTREND']
        )
    )
    
    return fig


def display_regime_stats(df_results):
    """Display regime statistics."""
    st.subheader("📊 Regime Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Current regime duration
    current_regime = df_results.iloc[-1]['state_name']
    regime_runs = (df_results['state_name'] != df_results['state_name'].shift()).cumsum()
    current_run_length = (regime_runs == regime_runs.iloc[-1]).sum()
    
    with col1:
        st.metric("Current Regime", current_regime)
    
    with col2:
        st.metric("Duration (bars)", current_run_length)
    
    with col3:
        avg_score = df_results.tail(20)['score'].abs().mean()
        st.metric("Avg Trend Strength (20 bars)", f"{avg_score:.3f}")
    
    with col4:
        volatility = df_results.tail(20)['atr_ratio'].mean()
        st.metric("Avg ATR Ratio (20 bars)", f"{volatility:.3f}")
    
    # Regime transition matrix
    st.subheader("🔄 Recent Regime Transitions")
    
    # Get last 200 bars
    recent = df_results.tail(200).copy()
    transitions = pd.crosstab(
        recent['state_name'].shift().fillna('START'),
        recent['state_name'],
        normalize='index'
    )
    
    if not transitions.empty:
        # Style the dataframe
        styled_transitions = transitions.style.format("{:.1%}").background_gradient(
            cmap='RdYlGn', axis=None
        )
        st.dataframe(styled_transitions, use_container_width=True)


def main():
    """Main Streamlit app."""
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Symbol and timeframe selection
    symbol = st.sidebar.selectbox(
        "Symbol",
        ["XAUUSDm", "EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm"],
        index=0
    )
    
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4
    }
    
    timeframe_str = st.sidebar.selectbox(
        "Timeframe",
        list(timeframe_map.keys()),
        index=2  # M15
    )
    timeframe = timeframe_map[timeframe_str]
    
    # Number of bars
    n_bars = st.sidebar.slider("Historical Bars", 100, 1000, 500, 50)
    
    # Display settings
    st.sidebar.subheader("📈 Display")
    chart_bars = st.sidebar.slider("Bars on Chart", 50, 500, 200, 50)
    
    # Auto-refresh
    st.sidebar.subheader("🔄 Auto Refresh")
    auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 60, 10)
    
    # Header
    st.title("📊 Market Regime Monitor")
    st.markdown(f"**Real-time regime detection for {symbol} {timeframe_str}**")
    
    # Load model
    with st.spinner("Loading regime model..."):
        if not load_predictor():
            st.error("❌ Failed to load regime model")
            st.stop()
    
    # Initialize MT5
    with st.spinner("Connecting to MT5..."):
        if not initialize_mt5(symbol, timeframe):
            st.error("❌ Failed to connect to MT5. Please ensure MT5 is running.")
            st.stop()
    
    # Fetch data button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    
    with col_btn1:
        if st.button("🔄 Fetch Data", type="primary"):
            with st.spinner("Fetching data from MT5..."):
                df = get_market_data(n_bars=n_bars)
                
                if df is None:
                    st.error("❌ Failed to fetch data from MT5")
                    st.stop()
                
                # Predict regimes
                with st.spinner("Analyzing market regime..."):
                    df_results = st.session_state.predictor.predict_history(df)
                    st.session_state.historical_data = df_results
                    st.session_state.last_update = datetime.now()
                
                st.success(f"✅ Loaded {len(df_results)} bars")
    
    with col_btn2:
        if st.button("📊 Refresh Display"):
            st.rerun()
    
    # Display last update time
    if st.session_state.last_update:
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main content
    if st.session_state.historical_data is not None:
        df_results = st.session_state.historical_data
        
        # Get latest prediction
        prediction = st.session_state.predictor.predict_latest(
            df_results[['open', 'high', 'low', 'close', 'tickvol', 'spread']]
        )
        
        if prediction:
            # Current status cards
            st.markdown("---")
            st.subheader("🎯 Current Market Status")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                regime_color = REGIME_COLORS.get(prediction['regime_now'], '#999999')
                st.markdown(
                    f"<div style='padding: 20px; background-color: {regime_color}; "
                    f"border-radius: 10px; text-align: center;'>"
                    f"<h3 style='color: white; margin: 0;'>{prediction['regime_now']}</h3>"
                    f"<p style='color: white; margin: 5px 0 0 0;'>Current</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                pred_regime = prediction['regime_in_1h_pred']
                pred_color = REGIME_3_COLORS.get(pred_regime, '#999999')
                st.markdown(
                    f"<div style='padding: 20px; background-color: {pred_color}; "
                    f"border-radius: 10px; text-align: center;'>"
                    f"<h3 style='color: white; margin: 0;'>{pred_regime}</h3>"
                    f"<p style='color: white; margin: 5px 0 0 0;'>In 1 Hour</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            with col3:
                st.metric(
                    "Trend Score",
                    f"{prediction['score']:.3f}",
                    delta=f"{prediction['trend_mom']:.3f}"
                )
            
            with col4:
                st.metric(
                    "Price",
                    f"{prediction['close']:.2f}",
                    delta=None
                )
            
            with col5:
                st.metric(
                    "ATR",
                    f"{prediction['atr']:.2f}",
                    delta=None
                )
            
            # Prediction probabilities
            st.markdown("---")
            st.subheader("🔮 1-Hour Regime Prediction")
            st.plotly_chart(
                plot_prediction_probabilities(prediction),
                use_container_width=True
            )
            
            # Direction probabilities
            col1, col2 = st.columns(2)
            with col1:
                up_prob = prediction['p_up_before_down']
                st.metric(
                    "📈 Up Before Down",
                    f"{up_prob:.1%}",
                    delta=f"{up_prob - 0.5:.1%} vs 50%"
                )
            
            with col2:
                down_prob = prediction['p_down_before_up']
                st.metric(
                    "📉 Down Before Up",
                    f"{down_prob:.1%}",
                    delta=f"{down_prob - 0.5:.1%} vs 50%"
                )
            
            # Main chart
            st.markdown("---")
            st.plotly_chart(
                plot_regime_chart(df_results, n_bars=chart_bars),
                use_container_width=True
            )
            
            # Two column layout for additional charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    plot_regime_distribution(df_results),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    plot_feature_importance(prediction),
                    use_container_width=True
                )
            
            # Regime timeline
            st.plotly_chart(
                plot_regime_timeline(df_results, n_bars=min(200, len(df_results))),
                use_container_width=True
            )
            
            # Statistics
            st.markdown("---")
            display_regime_stats(df_results)
            
            # Data table (expandable)
            with st.expander("📋 View Recent Data"):
                # Check which columns are available
                available_cols = []
                col_mapping = {}
                
                if 'close' in df_results.columns:
                    available_cols.append('close')
                    col_mapping['close'] = 'Close'
                if 'state_name' in df_results.columns:
                    available_cols.append('state_name')
                    col_mapping['state_name'] = 'Current Regime (5-class)'
                if 'regime_now_3class' in df_results.columns:
                    available_cols.append('regime_now_3class')
                    col_mapping['regime_now_3class'] = 'Current Regime (3-class)'
                if 'regime_pred_1h_name' in df_results.columns:
                    available_cols.append('regime_pred_1h_name')
                    col_mapping['regime_pred_1h_name'] = 'Predicted Regime (1h)'
                if 'score' in df_results.columns:
                    available_cols.append('score')
                    col_mapping['score'] = 'Score'
                if 'trend_mom' in df_results.columns:
                    available_cols.append('trend_mom')
                    col_mapping['trend_mom'] = 'Momentum'
                if 'atr' in df_results.columns:
                    available_cols.append('atr')
                    col_mapping['atr'] = 'ATR'
                if 'er16' in df_results.columns:
                    available_cols.append('er16')
                    col_mapping['er16'] = 'ER16'
                if 'adx' in df_results.columns:
                    available_cols.append('adx')
                    col_mapping['adx'] = 'ADX'
                
                recent_data = df_results[available_cols].tail(50).copy()
                recent_data.columns = [col_mapping[col] for col in available_cols]
                
                st.dataframe(
                    recent_data.style.format({
                        'Close': '{:.2f}',
                        'Score': '{:.3f}',
                        'Momentum': '{:.3f}',
                        'ATR': '{:.2f}',
                        'ER16': '{:.3f}',
                        'ADX': '{:.1f}'
                    }),
                    use_container_width=True
                )
        
        else:
            st.warning("⚠️ Could not generate prediction. Need more data bars.")
    
    else:
        st.info("👆 Click 'Fetch Data' to start monitoring")
    
    # Auto-refresh
    if auto_refresh and st.session_state.historical_data is not None:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == '__main__':
    main()
