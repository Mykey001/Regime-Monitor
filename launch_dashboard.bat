@echo off
echo ========================================
echo Market Regime Monitor Dashboard
echo ========================================
echo.
echo Starting Streamlit dashboard...
echo.
echo The dashboard will open in your browser.
echo Press Ctrl+C to stop the server.
echo.
echo ========================================

cd /d "%~dp0"
streamlit run streamlit_app.py

pause
