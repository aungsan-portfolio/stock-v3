@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ═══════════════════════════════════════════════
echo   🚀 BUY AMZN — Auto Setup
echo ═══════════════════════════════════════════════
echo.

:: Step 1: Patch MAX_OPEN_POSITIONS in config.py
echo [1/3] 📝 Updating MAX_OPEN_POSITIONS to 5...
python -X utf8 -c "
import re
path = r'config.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
content = re.sub(r'MAX_OPEN_POSITIONS\s*=\s*\d+', 'MAX_OPEN_POSITIONS = 5', content)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('   ✅ MAX_OPEN_POSITIONS = 5')
"
if %errorlevel% neq 0 (
    echo   ⚠️  Could not auto-update. Open config.py manually and change line 243:
    echo      MAX_OPEN_POSITIONS = 3  →  MAX_OPEN_POSITIONS = 5
)

:: Step 2: Dry-run preview first
echo.
echo [2/3] 🔮 Preview: checking AMZN signal...
python -X utf8 main.py paper --dry-run 2>&1 | findstr "AMZN\|BUY\|Orders accepted\|Skipped"
echo.

:: Step 3: Execute paper order
echo [3/3] 💼 Executing paper order for AMZN...
python -X utf8 main.py paper 2>&1 | findstr "AMZN\|Orders accepted\|Skipped\|BUY\|cleanup\|shutdown"
echo.
echo ═══════════════════════════════════════════════
echo   ✅ Done! Check TWS for AMZN position.
echo ═══════════════════════════════════════════════
echo.
pause
