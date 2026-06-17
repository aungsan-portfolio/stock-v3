@echo off
REM Phase 5B-1 supervised scheduler runner (Path A, PAPER ONLY).
REM Point Windows Task Scheduler at this .bat to recur it (e.g. every 30 min on
REM weekdays). The run-scheduled command is one-shot: it decides whether the
REM market is open and the scheduler is enabled, then runs `paper` in plan/dry-run
REM mode by default (no orders). It never loops and never enables live trading.
REM
REM PYTHONUTF8=1 (+ -X utf8) keeps the emoji-bearing logs from raising
REM UnicodeEncodeError under a non-UTF-8 scheduled console (H17 fix).
set PYTHONUTF8=1
cd /d "C:\Users\Aung San\Downloads\stock_engine_pro_v3_production_ready\stock_engine_pro_v3_production_ready"
python -X utf8 main.py run-scheduled >> logs\scheduled_run.log 2>&1
