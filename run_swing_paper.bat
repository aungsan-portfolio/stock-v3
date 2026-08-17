@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

REM ---------------------------------------------------------
REM 1. Clear credentials to prevent cross-account pollution
REM ---------------------------------------------------------
set "APCA_API_KEY_ID="
set "APCA_API_SECRET_KEY="
set "APCA_API_BASE_URL="

REM ---------------------------------------------------------
REM 2. Load Swing Credentials from .env safely
REM ---------------------------------------------------------
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "KEY_NAME=%%A"
        set "KEY_VAL=%%~B"
        if "!KEY_NAME!"=="SWING_APCA_API_KEY_ID" set "APCA_API_KEY_ID=!KEY_VAL!"
        if "!KEY_NAME!"=="SWING_APCA_API_SECRET_KEY" set "APCA_API_SECRET_KEY=!KEY_VAL!"
        if "!KEY_NAME!"=="SWING_APCA_API_BASE_URL" set "APCA_API_BASE_URL=!KEY_VAL!"
    )
) else (
    echo [ERROR] .env file not found. Please create .env from .env.example.
    pause
    exit /b 1
)

REM ---------------------------------------------------------
REM 3. Validation and Fail-Fast Guards
REM ---------------------------------------------------------
if not defined APCA_API_BASE_URL set "APCA_API_BASE_URL=https://paper-api.alpaca.markets"

REM Safety Guard: Ensure Paper API URL strictly
echo !APCA_API_BASE_URL! | findstr /i "paper-api.alpaca.markets" >nul
if %errorlevel% neq 0 (
    echo [CRITICAL ERROR] APCA_API_BASE_URL is NOT paper trading!
    echo Current URL: !APCA_API_BASE_URL!
    echo Refusing to execute swing bot to prevent real money losses.
    pause
    exit /b 1
)

REM Safety Guard: Check for missing or placeholder keys
if not defined APCA_API_KEY_ID (
    echo [ERROR] SWING_APCA_API_KEY_ID is not defined in .env!
    pause
    exit /b 1
)
if not defined APCA_API_SECRET_KEY (
    echo [ERROR] SWING_APCA_API_SECRET_KEY is not defined in .env!
    pause
    exit /b 1
)
echo !APCA_API_KEY_ID! | findstr /i "YOUR_" >nul
if %errorlevel% equ 0 (
    echo [ERROR] SWING_APCA_API_KEY_ID contains placeholder text. Please set real paper keys in .env.
    pause
    exit /b 1
)

REM ---------------------------------------------------------
REM 4. Lock File Tracking
REM ---------------------------------------------------------
set "LOCK_FILE=%~dp0data\swing_bot.lock"
if not exist "%~dp0data" mkdir "%~dp0data"
echo %DATE% %TIME% > "!LOCK_FILE!"

REM ---------------------------------------------------------
REM 5. Python Environment Selection
REM ---------------------------------------------------------
set "PY_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY_EXE=%~dp0.venv\Scripts\python.exe"

REM ---------------------------------------------------------
REM 6. Masked Display and Launch
REM ---------------------------------------------------------
set "MASKED_KEY=!APCA_API_KEY_ID:~0,4!****************!APCA_API_KEY_ID:~-4!"

echo =======================================================
echo   Starting Swing Trading Engine (Alpaca Swing Sub-Account)
echo   Account Key: !MASKED_KEY!
echo   Endpoint:    !APCA_API_BASE_URL!
echo   Python:      !PY_EXE!
echo =======================================================
echo.

"!PY_EXE!" -X utf8 main.py paper --broker alpaca --loop --interval 1800
set "EXIT_CODE=%errorlevel%"

REM Clean lock file on exit
if exist "!LOCK_FILE!" del "!LOCK_FILE!"

if %EXIT_CODE% neq 0 (
    echo.
    echo [ENGINE TERMINATED WITH ERROR CODE %EXIT_CODE%]
    pause
)
exit /b %EXIT_CODE%
