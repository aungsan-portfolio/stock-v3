@echo off
chcp 65001 >nul
set PYTHONUTF8=1

:: =========================================================
:: SWING TRADING ENGINE (IBKR PAPER / DRY-RUN)
:: Note: Swing Trading uses IBKR paper bridge or dry-run evaluation.
:: Multi-Account Safety Rule: Never run Day Trading and Swing Trading
:: on the same single Alpaca sub-account to prevent flatten_all() wipeouts.
:: =========================================================

cd /d "%~dp0"
echo =======================================================
echo   Starting Swing Trading Engine (Dry-Run / IBKR Paper)
echo =======================================================
echo.
python -X utf8 main.py paper --dry-run
pause
