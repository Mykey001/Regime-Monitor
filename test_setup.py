"""
Test Setup Script
=================

Verifies that all components are working correctly.
Run this before starting the dashboard.
"""

import sys
from pathlib import Path

print("=" * 70)
print("REGIME MONITOR - SETUP TEST")
print("=" * 70)
print()

# 1. Check Python version
print("1. Checking Python version...")
if sys.version_info < (3, 8):
    print("   ❌ Python 3.8+ required")
    sys.exit(1)
else:
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print()

# 2. Check dependencies
print("2. Checking dependencies...")
required_packages = [
    'streamlit',
    'pandas',
    'numpy',
    'plotly',
    'MetaTrader5',
    'joblib',
    'sklearn',
    'lightgbm',
    'matplotlib'
]

missing = []
for package in required_packages:
    try:
        __import__(package if package != 'sklearn' else 'sklearn')
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ❌ {package} not found")
        missing.append(package)

if missing:
    print()
    print("Missing packages. Install with:")
    print("   pip install -r requirements.txt")
    print()
else:
    print()

# 3. Check model file
print("3. Checking model file...")
model_path = Path(__file__).parent.parent / "models" / "reversion_bundle.joblib"
if model_path.exists():
    print(f"   ✓ Model found: {model_path}")
    
    # Try to load it
    try:
        import joblib
        bundle = joblib.load(model_path)
        print(f"   ✓ Model loaded successfully")
        print(f"   ℹ Model type: {bundle['config']['model']}")
        print(f"   ℹ Features: {len(bundle['config']['FEATS'])}")
        print(f"   ℹ Trained through: {bundle['config']['trained_through']}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
else:
    print(f"   ❌ Model not found: {model_path}")
    print("   Run the training notebook (Reversion_regime_model.ipynb) first")
print()

# 4. Check MT5 connection
print("4. Checking MT5 connection...")
try:
    import MetaTrader5 as mt5
    
    if mt5.initialize():
        print("   ✓ MT5 initialized")
        
        # Check account
        account_info = mt5.account_info()
        if account_info:
            print(f"   ✓ Connected to account: {account_info.login}")
            print(f"   ℹ Server: {account_info.server}")
            print(f"   ℹ Balance: {account_info.balance:.2f} {account_info.currency}")
        
        # Check XAUUSD availability
        symbol_info = mt5.symbol_info("XAUUSD")
        if symbol_info:
            print(f"   ✓ XAUUSD available")
        else:
            print(f"   ⚠ XAUUSD not found (you can use other symbols)")
        
        # Get a test bar
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 10)
        if rates is not None:
            print(f"   ✓ Can fetch historical data ({len(rates)} bars)")
        
        mt5.shutdown()
    else:
        print("   ❌ Failed to initialize MT5")
        print("   Make sure MetaTrader 5 is running")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# 5. Check directories
print("5. Checking directories...")
directories = ['data', 'logs', 'cache']
for dir_name in directories:
    dir_path = Path(__file__).parent / dir_name
    if not dir_path.exists():
        dir_path.mkdir(exist_ok=True)
        print(f"   ✓ Created: {dir_name}/")
    else:
        print(f"   ✓ Exists: {dir_name}/")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)

if missing:
    print("❌ Setup incomplete - missing packages")
    print("   Run: pip install -r requirements.txt")
elif not model_path.exists():
    print("❌ Setup incomplete - model not found")
    print("   Run the training notebook first")
else:
    print("✅ Setup complete!")
    print()
    print("To start the dashboard:")
    print("   streamlit run streamlit_app.py")
    print()
    print("Or double-click:")
    print("   launch_dashboard.bat")

print("=" * 70)
