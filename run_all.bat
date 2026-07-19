@echo off
chcp 65001 >nul
cd /d "%~dp0"

set LOGFILE=logs\run_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.txt
echo ============================================ > "%LOGFILE%"
echo   Stock Engine Pro V3 — Full Auto Run       >> "%LOGFILE%"
echo   %date% %time%                              >> "%LOGFILE%"
echo ============================================ >> "%LOGFILE%"

echo ╔══════════════════════════════════════════════╗
echo ║   STOCK ENGINE PRO V3 — FULL AUTO PIPELINE  ║
echo ║   %date% %time%                 ║
echo ╚══════════════════════════════════════════════╝
echo.

:: ── Step 1: Predict WATCHLIST ──
echo [1/6] 🔮 Predicting WATCHLIST signals...
echo --- STEP 1: Predict WATCHLIST --- >> "%LOGFILE%"
python -X utf8 main.py predict >> "%LOGFILE%" 2>&1
if %errorlevel% equ 0 (echo   ✅ Done) else (echo   ⚠️  Check log)
echo.

:: ── Step 2: Hot Scanner ──
echo [2/6] 🔍 Scanning market for hot candidates...
echo --- STEP 2: Hot Scanner --- >> "%LOGFILE%"
python -X utf8 main.py scan-hot --full-market --max-symbols 200 --top-n 50 >> "%LOGFILE%" 2>&1
if %errorlevel% equ 0 (echo   ✅ Done) else (echo   ⚠️  Check log)
echo.

:: ── Step 3: Predict Hot ──
echo [3/6] 🔮 Predicting hot candidates...
echo --- STEP 3: Predict Hot --- >> "%LOGFILE%"
python -X utf8 main.py predict-hot --full-market --max-symbols 200 --top-n 50 >> "%LOGFILE%" 2>&1
if %errorlevel% equ 0 (echo   ✅ Done) else (echo   ⚠️  Check log)
echo.

:: ── Step 4: Model Doctor ──
echo [4/6] 🏥 Checking model health...
echo --- STEP 4: Model Doctor --- >> "%LOGFILE%"
python -X utf8 main.py model-doctor >> "%LOGFILE%" 2>&1
if %errorlevel% equ 0 (echo   ✅ Done) else (echo   ⚠️  Check log)
echo.

:: ── Step 5: Coach (WATCHLIST lessons) ──
echo [5/6] 🎓 Getting trading coach lessons...
echo --- STEP 5: Coach --- >> "%LOGFILE%"
python -X utf8 main.py coach >> "%LOGFILE%" 2>&1
if %errorlevel% equ 0 (echo   ✅ Done) else (echo   ⚠️  Check log)
echo.

:: ── Step 6: Print Summary ──
echo [6/6] 📋 Generating summary...
echo.
echo ╔══════════════════════════════════════════════╗
echo ║   📋 TODAY'S SUMMARY                        ║
echo ╚══════════════════════════════════════════════╝
echo.
python -X utf8 -c "
import re
log = open(r'%LOGFILE%', encoding='utf-8').read()

# Extract BUY signals
buys = re.findall(r'(?:(?:\d{4}-\d{2}-\d{2}.*?predictor.*?)|(?:\n))(\w{1,6})\s+BUY\s+conf=([\d.]+)', log)
if buys:
    print('🟢  BUY SIGNALS FOUND:')
    for sym, conf in buys:
        print(f'    {sym:<6}  conf={conf}')
else:
    print('🟡  No BUY signals today (HOLD/SELL only)')
    print('     WATCHLIST top HOLD near threshold:')
    holds = re.findall(r'(\w{1,6})\s+HOLD\s+conf=([\d.]+)', log)
    holds = sorted(holds, key=lambda x: float(x[1]), reverse=True)[:3]
    for sym, conf in holds:
        print(f'    {sym:<6}  conf={conf}')

# Extract hot candidates
hots = re.findall(r'Top 10 hot candidates:.*?(?:\n\s+)(.*)', log)
if hots:
    print()
    print('🔥  Hot Candidates:')
    for h in hots:
        symbols = re.findall(r'(\w+)', h)
        if symbols:
            print(f'    {", ".join(symbols[:5])}')

print()
print(f'📄  Full log: {r"%LOGFILE%"}')
"
echo.
echo ══════════════════════════════════════════════
echo   ✅ Pipeline complete! Share the output with
echo      Claude to get BUY recommendations.
echo ══════════════════════════════════════════════
echo.
pause
