@echo off
chcp 65001 >nul
set PYTHONUTF8=1

:: =========================================================
:: DAY TRADING ALPACA PAPER CREDENTIALS
:: =========================================================
cd /d "%~dp0"
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if "%%A"=="DAYTRADE_APCA_API_KEY_ID" set APCA_API_KEY_ID=%%B
        if "%%A"=="DAYTRADE_APCA_API_SECRET_KEY" set APCA_API_SECRET_KEY=%%B
        if "%%A"=="DAYTRADE_APCA_API_BASE_URL" set APCA_API_BASE_URL=%%B
    )
) else (
    echo [NOTICE] .env file not found. Loading credentials from environment...
)

if not defined APCA_API_KEY_ID set APCA_API_KEY_ID=YOUR_DAYTRADE_ALPACA_API_KEY
if not defined APCA_API_SECRET_KEY set APCA_API_SECRET_KEY=YOUR_DAYTRADE_ALPACA_SECRET_KEY
if not defined APCA_API_BASE_URL set APCA_API_BASE_URL=https://paper-api.alpaca.markets

echo =======================================================
echo   Starting Day Trading Engine (Alpaca Paper Account)
echo =======================================================
echo.
python -X utf8 main.py daytrade-bot --live-paper
pause
