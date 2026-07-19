@echo off
chcp 65001 >nul
cd /d "%~dp0"
cls
echo ============================================
echo   Stock Engine Pro V3 — Phone Access Setup
echo ============================================
echo.
echo Starting Dashboard Server...
echo.
start /b "" python -X utf8 dashboard.py
timeout /t 3 /nobreak >nul
cls
echo ============================================
echo   Stock Engine Pro V3 Dashboard
echo ============================================
echo.
python -c "
import socket, qrcode

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
ip = s.getsockname()[0]
s.close()

port = 5050
url = f'http://{ip}:{port}'

print(f'  Your phone access URL:')
print(f'  {url}')
print()
qr = qrcode.QRCode(border=2, box_size=2)
qr.add_data(url)
qr.print_ascii(invert=True)
print()
print(f'  [1] Phone must be on SAME WiFi network')
print(f'  [2] Open browser and type the URL above')
print(f'  [3] Or scan QR code with phone camera')
print()
print(f'  Press Ctrl+C in this window to stop')
"
echo.
echo Dashboard is running in background.
echo Press any key to STOP the server...
pause >nul
taskkill /f /im python.exe 2>nul >nul
cls
echo Dashboard stopped.
timeout /t 2 /nobreak >nul
