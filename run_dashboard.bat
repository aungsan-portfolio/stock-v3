@echo off
chcp 65001 >nul
echo ============================================
echo   Stock Engine Pro V3 — GUI Dashboard
echo ============================================
echo.
echo Starting dashboard server...
echo Open http://localhost:5050 in your browser
echo Press Ctrl+C to stop
echo.
python -X utf8 dashboard.py
pause
