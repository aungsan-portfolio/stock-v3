@echo off
chcp 65001 >nul
set PYTHONUTF8=1

:: =========================================================
:: DAY TRADING ALPACA PAPER CREDENTIALS
:: Replace YOUR_DAYTRADE_ALPACA_API_KEY and YOUR_DAYTRADE_ALPACA_SECRET_KEY below
:: =========================================================
set APCA_API_KEY_ID=PKMEKPN5QXHM5QRGKWMIQGDQDD
set APCA_API_SECRET_KEY=Fnh7DDi3AV2spLCACm8CJh1ksjaworq7ig84oPReP4hn
set APCA_API_BASE_URL=https://paper-api.alpaca.markets

cd /d "%~dp0"
echo =======================================================
echo   Starting Day Trading Engine (Alpaca Paper Account)
echo =======================================================
echo.
python -X utf8 main.py daytrade-bot --live-paper
pause
