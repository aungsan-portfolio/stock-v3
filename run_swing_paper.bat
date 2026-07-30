@echo off
chcp 65001 >nul
set PYTHONUTF8=1

:: =========================================================
:: SWING TRADING ALPACA PAPER CREDENTIALS (ISOLATED SUB-ACCOUNT)
:: Replace YOUR_SWING_ALPACA_API_KEY and YOUR_SWING_ALPACA_SECRET_KEY below
:: Multi-Account Safety Rule: Day Trading and Swing Trading MUST run on
:: separate Alpaca Sub-accounts to keep positions isolated.
cd /d "%~dp0"
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if "%%A"=="SWING_APCA_API_KEY_ID" set APCA_API_KEY_ID=%%B
        if "%%A"=="SWING_APCA_API_SECRET_KEY" set APCA_API_SECRET_KEY=%%B
        if "%%A"=="SWING_APCA_API_BASE_URL" set APCA_API_BASE_URL=%%B
    )
)
if not defined APCA_API_KEY_ID set APCA_API_KEY_ID=YOUR_SWING_ALPACA_API_KEY
if not defined APCA_API_SECRET_KEY set APCA_API_SECRET_KEY=YOUR_SWING_ALPACA_SECRET_KEY
if not defined APCA_API_BASE_URL set APCA_API_BASE_URL=https://paper-api.alpaca.markets
echo =======================================================
echo   Starting Swing Trading Engine (Alpaca Swing Sub-Account)
echo =======================================================
echo.
python -X utf8 main.py paper --broker alpaca --loop --interval 1800
pause
