@echo off
chcp 65001 >nul
set PYTHONUTF8=1

:: =========================================================
:: SWING TRADING ALPACA PAPER CREDENTIALS (ISOLATED SUB-ACCOUNT)
:: Replace YOUR_SWING_ALPACA_API_KEY and YOUR_SWING_ALPACA_SECRET_KEY below
:: Multi-Account Safety Rule: Day Trading and Swing Trading MUST run on
:: separate Alpaca Sub-accounts to keep positions isolated.
:: =========================================================
set APCA_API_KEY_ID=YOUR_SWING_ALPACA_API_KEY
set APCA_API_SECRET_KEY=YOUR_SWING_ALPACA_SECRET_KEY
set APCA_API_BASE_URL=https://paper-api.alpaca.markets

cd /d "%~dp0"
echo =======================================================
echo   Starting Swing Trading Engine (Alpaca Swing Sub-Account)
echo =======================================================
echo.
python -X utf8 main.py paper --broker alpaca
pause
