@echo off
chcp 65001 >nul
cd /d "%~dp0"
cls
echo ═══════════════════════════════════════════
echo   🔍 STOCK ENGINE — BUY SIGNAL CHECKER
echo   %date% %time%
echo ═══════════════════════════════════════════
echo.

:: Step 1: Predict WATCHLIST
echo [1/3] 🔮 Predicting WATCHLIST...
python -X utf8 main.py predict 2>&1 | findstr /r "BUY\|HOLD\|Summary\|^\w"

echo.
:: Step 2: Scan + Predict Hot (fast --full-market 200)
echo [2/3] 🔍 Scanning + Predicting hot candidates...
python -X utf8 main.py predict-hot --full-market --max-symbols 200 --top-n 10 2>&1 | findstr /r "BUY\|HOLD\|SELL\|Summary\|^$"

echo.
echo [3/3] 📋 Checking for BUY opportunities...
python -X utf8 -c "
import re, sys
try:
    text = open('logs\\buy_check_temp.txt', encoding='utf-8').read() if False else ''
except: pass

# Re-read predict output
print('═══ WATCHLIST ═══')
from predictor import Predictor
from config import BUY_THRESHOLD
sigs = Predictor().predict_all()
buys = [s for s in sigs if s.action == 'BUY']
holds = [s for s in sigs if s.action == 'HOLD']
if buys:
    print(f'🟢  BUY SIGNALS FOUND ({len(buys)}):')
    for s in buys:
        print(f'    {s.symbol:<6}  conf={s.confidence:.2f}  price=\${s.price:.2f}')
else:
    print(f'🟡  No BUY signals today (threshold={BUY_THRESHOLD})')
    print(f'     Closest to BUY:')
    near_holds = sorted(holds, key=lambda s: s.confidence, reverse=True)[:5]
    for s in near_holds:
        dist = (BUY_THRESHOLD - s.confidence) * 100
        print(f'    {s.symbol:<6}  conf={s.confidence:.2f}  (needs +{dist:.0f}%)')
" 2>&1

echo.
echo ═══════════════════════════════════════════
echo   ✅ Done! Share output with Claude.
echo ═══════════════════════════════════════════
echo.
pause
