"""
config.py — Stock Prediction Engine Configuration
Single source of truth — no magic numbers anywhere else.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
MODELS_DIR      = BASE_DIR / "models"
REPORTS_DIR     = BASE_DIR / "reports"
LOG_DIR         = BASE_DIR / "logs"

ML_MODELS_FILE  = MODELS_DIR / "rf_models.joblib"
LSTM_CKPT_FILE  = MODELS_DIR / "lstm_checkpoint.pt"

# ── Stock Universe ───────────────────────────────
WATCHLIST = [
    "SPY", "QQQ", "VTI",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
]

# ── Full-Market Hot Scanner ──────────────────────
# Safe, opt-in broad market discovery. This ONLY produces candidate symbols.
# Prediction + risk rules still decide BUY/HOLD/SELL and whether any paper
# order is placed. Nothing here enables live trading or auto-buying.
FULL_MARKET_SCAN_ENABLED        = True
# How long the downloaded Nasdaq Trader symbol directory is cached before a
# re-download is attempted (data/symbol_universe.csv).
FULL_MARKET_CACHE_HOURS         = 24
# Hard cap on how many symbols a single full-market scan will fetch OHLCV for.
# Beginner-safe default; raise deliberately once you understand the runtime cost.
FULL_MARKET_MAX_SYMBOLS_TO_CHECK = 500

# ── Full-market symbol SELECTION (which symbols out of the universe to scan) ──
# The universe is stored alphabetically. Taking the first N would scan only A/AB
# tickers, so selection decides *which* slice of the broad market each run looks
# at. This is discovery only — it never places orders or changes risk rules.
#   "alphabetical" : first N symbols (original behavior; for debugging only)
#   "random"       : seeded random sample from the whole universe
#   "rotation"     : a different sequential slice each run (covers the market over time)
#   "hybrid"       : always include the core symbols below, then fill with a
#                    rotating, shuffled sample of the rest (DEFAULT)
FULL_MARKET_SELECTION_MODE = "hybrid"

# Anchors always scanned in "hybrid" mode (when present in the universe) so a
# broad scan never loses sight of the most liquid, widely-followed names.
FULL_MARKET_CORE_SYMBOLS = [
    "SPY", "QQQ", "VTI",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN",
    "GOOGL", "AVGO", "NFLX", "COST", "JPM", "BAC", "XOM", "UNH",
]

# Seed for the "random" sample and the hybrid fill shuffle — fixed for
# reproducible selections; change it to draw a different sample.
FULL_MARKET_RANDOM_SEED = 42

# Persisted rotation cursor so "rotation"/"hybrid" advance through the universe
# across runs instead of always starting in the same place.
FULL_MARKET_ROTATION_STATE_FILE = DATA_DIR / "scan_rotation_state.json"
# Keep only this many top-ranked candidates after scoring.
HOT_SCAN_TOP_N                  = 30
# Price band: skip penny stocks and very expensive tickers.
HOT_SCAN_MIN_PRICE              = 5.0
HOT_SCAN_MAX_PRICE              = 1000.0
# Liquidity floor: skip thinly traded names.
HOT_SCAN_MIN_AVG_VOLUME         = 1_000_000
# When True, ETFs are excluded from candidates (common stocks only).
HOT_SCAN_EXCLUDE_ETFS           = False
# Reject names whose ATR% (volatility) is above this fraction of price.
HOT_SCAN_MAX_ATR_PCT            = 0.12
# Batch sizing + polite throttling for the data fetch loop.
HOT_SCAN_CHUNK_SIZE             = 50
HOT_SCAN_SLEEP_SECONDS          = 1.0

# Nasdaq Trader official symbol directory files (pipe-delimited text).
NASDAQ_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL   = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
SYMBOL_UNIVERSE_FILE = DATA_DIR / "symbol_universe.csv"
HOT_CANDIDATES_FILE  = REPORTS_DIR / "hot_candidates.csv"
# Audit trail of exactly which symbols a scan selected (and why) before scanning.
SELECTED_SCAN_SYMBOLS_FILE = REPORTS_DIR / "selected_scan_symbols.csv"

# ── IPO / New Listing Watch (WATCH-ONLY) ─────────
# Symbols tracked for visibility only. daily-coach always shows a separate
# "IPO / New Listing Watch" section for these. They are NEVER auto-trained,
# NEVER auto-traded, and NEVER placed as orders. A symbol here only becomes an
# official BUY candidate if it independently passes the normal model/risk rules
# (i.e. it is also discovered/scanned and the ensemble issues a BUY). New
# listings with fewer than MIN_HISTORY_DAYS of history are flagged watch-only.
IPO_WATCH_SYMBOLS = ["SPCX"]

# ── Data Settings ────────────────────────────────
PRICE_PERIOD        = "5y"
PRICE_INTERVAL      = "1d"
MIN_HISTORY_DAYS    = 252
CACHE_TTL_SECONDS   = 3600

# ── Technical Indicators ─────────────────────────
SMA_SHORT           = 20
SMA_LONG            = 50
EMA_PERIOD          = 21
RSI_PERIOD          = 14
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
BOLLINGER_PERIOD    = 20
BOLLINGER_STD       = 2
ATR_PERIOD          = 14
VOLUME_MA_PERIOD    = 20

# ── ML Settings ─────────────────────────────────
ML_WINDOW           = 20
ML_HORIZON          = 5
ML_TEST_RATIO       = 0.15
RANDOM_STATE        = 42
RF_N_ESTIMATORS     = 100
RF_MAX_DEPTH        = 6

# ── LSTM Settings ────────────────────────────────
LSTM_HIDDEN         = 64
LSTM_LAYERS         = 2
LSTM_DROPOUT        = 0.3
LSTM_EPOCHS         = 50
LSTM_BATCH          = 32
LSTM_LR             = 1e-3
LSTM_PATIENCE       = 7
LSTM_WINDOW         = 30

# ── Signal Thresholds ────────────────────────────
# Higher BUY threshold = fewer entries, better fit for risk-controlled trading.
BUY_THRESHOLD       = 0.65
SELL_THRESHOLD      = 0.40

# Minimum bars a position is held before an opposite signal may close it.
# Aligned with ML_HORIZON so the held period matches the forward-direction
# label the models are trained on.
MIN_HOLD_BARS       = ML_HORIZON

# ── Ensemble Weights ─────────────────────────────
WEIGHT_RF           = 0.40
WEIGHT_LSTM         = 0.35
WEIGHT_TECHNICAL    = 0.25

# Trading safety: do not allow technical-only BUY/SELL signals.
# 1 = at least one ML model (RF or LSTM) must be available; otherwise forced HOLD.
MIN_ML_MODELS_FOR_SIGNAL = 1

# ── Phase 1: Model performance / staleness gate (signal safety) ───
# Per-symbol RF/LSTM metrics are persisted here at train time (see model_metrics.py)
# and read by predictor.predict_all before issuing any BUY/SELL. A symbol whose
# metrics are MISSING, STALE, or BELOW the floor is forced to HOLD with an
# explanation. This is fail-closed by design: no trusted metric => no trade.
MODEL_METRICS_FILE   = MODELS_DIR / "model_metrics.json"
# Master switch for the gate. True = enforce (recommended). Tests may disable it
# to isolate ensemble logic; do not ship it False.
MODEL_GATE_ENABLED   = True
# Minimum cross-validated RF accuracy required to trade a symbol. 0.50 = at least
# coin-flip; a model at or below chance must not place orders. Raise to demand a
# real edge once models clear this bar.
MODEL_MIN_RF_ACCURACY = 0.50
# Optional RF F1 floor. 0.0 = disabled; raise to also require balanced precision/recall.
MODEL_MIN_RF_F1       = 0.0
# Maximum model age (days since training) before a symbol is forced to HOLD.
# Regime shifts make stale models dangerous; retrain to refresh the timestamp.
MODEL_MAX_AGE_DAYS    = 30

# ── Risk Management ──────────────────────────────
# Small-basket mode: allow a few tiny positions instead of one large holding.
MAX_POSITION_PCT    = 0.005
# Absolute dollar cap per trade. Final size = min(cash * MAX_POSITION_PCT,
# MAX_TRADE_VALUE) so a large account never sizes a single trade above this.
MAX_TRADE_VALUE     = 500.0
MAX_OPEN_POSITIONS  = 3

# Fast loss control. Kept small because the goal is to be wrong small.
STOP_LOSS_PCT       = 0.004   # -0.40% initial protective stop
TAKE_PROFIT_PCT     = 0.015   # fallback fixed-bracket TP when trailing is disabled

# Minimum forward return required for a BUY label. Set above broker fee + slippage
# round trip so the model learns to cover costs, not microscopic moves.
MIN_PROFIT_MARGIN   = 0.003
# Absolute backstop exit. A hard stop that overrides trailing logic to cap the
# worst-case loss on any single position regardless of other exit settings.
HARD_STOP_LOSS_PCT  = 0.03

# Winner management. Native trailing stop lets a correct trade keep running while
# protecting profit when price reverses from its high.
USE_TRAILING_EXIT   = True
TRAILING_STOP_PCT   = 0.004   # fraction; IBKR trailingPercent uses this * 100 = 0.4%

# ── Production Safety Guards ─────────────────────
# Default is long-only. SELL closes existing long positions; it does not open shorts.
ALLOW_SHORT          = False
MIN_TRADE_CASH       = 100.0
LIMIT_ORDER_OFFSET_PCT = 0.001

# Daily safety caps. These are intentionally conservative for paper-testing.
MAX_DAILY_TRADES     = 6
MAX_DAILY_LOSS_USD   = 150.0

# Do not use yesterday's/daily historical close as the price source for order placement.
ALLOW_HISTORICAL_PRICE_FOR_ORDERS = False

# ── Phase 2: Order execution robustness (H4-H9) ──
# Fill-driven confirmation replaces the old fixed ib.sleep(1) "accepted" check.
# How long to wait for an order to reach a terminal (Filled/Cancelled) state
# before classifying it WORKING/TIMEOUT, and how often to poll the event loop
# while waiting. Kept short so a one-shot run does not hang; a timed-out order
# still rests at the broker and is reconciled by the duplicate/working-order
# guards on the next run.
ORDER_FILL_TIMEOUT_SECONDS = 8.0
ORDER_POLL_SECONDS         = 0.25
# Robust close: how many escalating attempts (marketable-limit -> market) to make
# to drive a close to remaining == 0 before giving up and alerting.
CLOSE_MAX_ATTEMPTS         = 3

# ── Phase 3: Server-side protection + risk engine (C2, H19, H1, H3) ──
# Protective exits rest at the broker as GTC so they survive between the
# one-shot bot's runs (and overnight). All exits for one position share an OCA
# group so the first to fill auto-cancels the rest (no leftover resting SELL
# that could open an accidental short). HARD_STOP_LOSS_PCT (above) is the
# independent catastrophe stop, sized to the actual filled qty.
PROTECTIVE_TIF             = "GTC"
# Account circuit breakers (H3 GROUNDWORK only; conservative paper-test values).
# Block NEW entries when equity falls this fraction below the start-of-day
# snapshot. Closes / flatten / protective repair stay allowed.
ACCOUNT_MAX_DRAWDOWN_PCT   = 0.10
# Per-symbol exposure cap as a fraction of current equity (groundwork). A single
# new position whose value exceeds this share of equity is refused.
MAX_SYMBOL_EXPOSURE_PCT    = 0.10

# ── Phase 4: Data integrity for order pricing (H11-H16) ──
# Guards that the price an order is placed at is real-time-enough, internally
# sane, and agrees with the price the decision was made on. PAPER-SAFE: these
# only refuse a NEW entry on a bad/stale quote; they never enable live trading,
# never touch closes/flatten/protective repair, and keep delayed-data paper
# trading working (REQUIRE_REALTIME_DATA_FOR_ORDERS stays False until live).
#
# When True, an order may ONLY be priced from a real-time last/midpoint; delayed
# and prior-close fields are refused. This MUST stay False for paper accounts on
# the delayed (15-min) feed (IBKR_MARKET_DATA_TYPE=3) or no order would ever get
# a price. It is flipped True only at live conversion (Phase 6), together with a
# real-time market-data subscription. The crossed/wide-spread/stale-close sanity
# checks run regardless of this flag.
REQUIRE_REALTIME_DATA_FOR_ORDERS = False
# Reject a quote whose bid/ask spread exceeds this fraction of the midpoint (an
# illiquid / unreliable snapshot). 0 disables the check.
MAX_QUOTE_SPREAD_PCT             = 0.02   # 2%
# Refuse to act on a signal when the live order price has moved more than this
# fraction from the price the decision was made on (a stale / gapped signal,
# H14). In this bot the decision price is a daily close and the order price is
# an intraday quote, so this is a LARGE-GAP guard; tighten it toward 0.005 only
# once decisions are made on the live IBKR quote. 0 disables the check.
DECISION_PRICE_MAX_DEVIATION_PCT = 0.05   # 5%
# US regular-hours gate for NEW entries (H15). True = refuse to OPEN a new
# position outside 09:30-16:00 ET on a trading day (closes, emergency flatten,
# and protective repair are NEVER gated). Set False to practise paper entries
# outside market hours. Uses an approximate computed US holiday/early-close
# calendar (data_integrity.py) -- not the broker calendar.
MARKET_HOURS_GATE_ENABLED        = True

# ── Backtest Settings ────────────────────────────
# Daily position-state simulation costs. These are conservative defaults.
BACKTEST_TRANSACTION_COST_PCT = 0.0005  # 5 bps per order
BACKTEST_SLIPPAGE_PCT         = 0.0005  # 5 bps per order

# Full walk-forward LSTM inside backtest is expensive. Default backtest uses
# walk-forward RF + technical score with the same BUY/SELL thresholds and
# broker-like position-state rules. Enable this only for slower full-ensemble runs.
BACKTEST_INCLUDE_LSTM         = False
BACKTEST_LSTM_EPOCHS          = 8
BACKTEST_LSTM_BATCH           = 64

# ── IBKR Paper Trading ───────────────────────────
IBKR_HOST           = "127.0.0.1"
IBKR_PORT           = 7497
PAPER_IBKR_PORT     = 7497
REQUIRE_PAPER_PORT  = True

CLIENT_ID_BOT       = 1
CLIENT_ID_CHECK     = 11
CLIENT_ID_CANCEL    = 12
CLIENT_ID_FLATTEN   = 13

# Backward-compatible name used by older code.
IBKR_CLIENT_ID      = CLIENT_ID_BOT

PAPER_CAPITAL       = 100_000.0
# Market data type: 1=live, 2=frozen, 3=delayed (15-min), 4=delayed-frozen.
# 3 lets paper accounts without a real-time subscription still get prices.
IBKR_MARKET_DATA_TYPE = 3

# ── Logging ──────────────────────────────────────
LOG_FILE            = LOG_DIR / "stock_engine.log"
LOG_MAX_BYTES       = 10 * 1024 * 1024
LOG_BACKUP_COUNT    = 5


# ─── Guided Paper Trading Coach ─────────────────────────────────────────────
# A read-mostly, beginner-friendly flow that turns ensemble signals into a
# trade lesson + paper preview, and only places a paper order when the user
# explicitly confirms with BOTH `--confirm` AND `--chart-checked`.
# Default behavior is preview only; nothing here auto-buys.
COACH_MIN_CONFIDENCE_FOR_CANDIDATE = 0.65
COACH_REQUIRE_CHART_CHECK          = True
COACH_REQUIRE_USER_CONFIRM         = True
# Cap how many new trades a single `paper-coach` run can propose. That older
# single-symbol flow is meant to teach, not to generate a batch of orders.
COACH_MAX_NEW_TRADES_PER_RUN       = 1
# Where the human-readable lesson / preview is written. Always written, even
# when no order is placed, so the user has an audit trail.
COACH_REPORT_FILE                  = REPORTS_DIR / "trade_coach_report.md"
# When True, the coach refuses to propose a paper trade for any symbol that is
# already in positions() or has a working order on the paper account.
COACH_SKIP_IF_POSITION_OR_WORKING  = True

# ─── Guided Daily Trading Coach (multi-trade PAPER practice) ────────────────
# The `daily-coach` command scans the full market, previews the best BUY
# candidates, and — only with BOTH --confirm AND --chart-checked — may place
# up to COACH_MAX_PAPER_TRADES_PER_RUN *paper* orders in a single run so the
# user can practice and learn faster. Every existing risk control still
# applies (MAX_TRADE_VALUE, MAX_POSITION_PCT, MAX_OPEN_POSITIONS,
# MAX_DAILY_TRADES, ALLOW_SHORT=False, REQUIRE_PAPER_PORT=True, the
# duplicate-position/working-order guards, and live-snapshot-only pricing).
#
# Hard cap on how many PAPER trades one daily-coach run may place. This is an
# upper bound: --max-trades can only LOWER it, never raise it.
COACH_MAX_PAPER_TRADES_PER_RUN     = 3
# How many top candidates the default (preview-only) daily-coach run shows.
COACH_DEFAULT_PREVIEW_CANDIDATES   = 3
# Master live-trading kill switch. This bot is PAPER-TRADING ONLY. This must
# stay False; daily-coach refuses to run if anything tries to flip it True or
# tries to connect on a non-paper port while REQUIRE_PAPER_PORT is True.
COACH_LIVE_TRADING_ENABLED         = False

# ─── Phase 5B-1: Supervised Scheduler / Market-Hours Runner (Path A) ─────────
# A THIN orchestration layer (`scheduler_runner.py`, command `run-scheduled`) for
# the one-shot Path A bot. It only DECIDES whether a scheduled run is allowed and
# then calls the EXISTING one-shot paper command. It adds NO trading logic, NO
# daemon loop, and NO new capability flag. Every existing gate still applies
# (model gate, data-integrity gate, market-hours gate, daily-loss kill-switch,
# startup reconciliation, paper-port lock). PAPER ONLY: it never enables live
# trading and ships in plan/dry-run mode.
#
# For Windows Task Scheduler use `run_scheduled.bat` and set PYTHONUTF8=1 (or
# `python -X utf8`) so the emoji-bearing logs cannot raise UnicodeEncodeError in
# a non-UTF-8 scheduled console (the H17 `scheduled_dryrun.log` fix).
#
# Master switch for the `run-scheduled` command. False => every scheduled run is
# blocked (logged + audited) and nothing is dispatched.
SCHEDULER_ENABLED            = True
# Default execution mode. True (ship value) => even `run-scheduled --execute`
# stays a dry-run/plan preview and places NO orders. Flip to False ONLY when you
# deliberately want `--execute` to forward to the (paper) order path; the bot is
# still paper-locked, so this can never place a live order.
SCHEDULER_DRY_RUN_DEFAULT    = True
# Require US regular trading hours (RTH) for a scheduled run. True (recommended)
# => a run outside 09:30-16:00 ET on a trading day (weekend/holiday/early-close
# aware) is blocked, mirroring the Phase-4 market-hours gate via the same
# data_integrity calendar. Set False only to practise scheduled plan runs
# outside market hours.
SCHEDULER_REQUIRE_RTH        = True

# ─── Phase 5B-2: Reconnect watchdog / connection resilience (Path A) ─────────
# Connection hardening for the ONE-SHOT Path A bot (see reconnect_watchdog.py and
# reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md task 5.3). This is NOT a daemon
# and NOT a forever reconnect loop. It only:
#   * wraps the INITIAL connect in a BOUNDED exponential-backoff retry, so a
#     transient TWS hiccup at startup does not abort the whole one-shot run; and
#   * registers an ib.disconnectedEvent handler that, on a mid-run drop, marks
#     the connection UNHEALTHY and FAILS CLOSED (no new entries) — it never
#     places, cancels, or modifies any order.
# It adds NO new live-readiness capability flag and NEVER bypasses a safety gate:
# the paper-port lock is enforced BEFORE any retry, and startup reconciliation,
# the data-integrity gate, the market-hours gate, the daily-loss kill-switch, and
# the model gate all still run unchanged. PAPER ONLY throughout.
#
# Master switch. False => connect() behaves exactly as before (a single attempt,
# no retry). The disconnect handler still marks health, but no reconnect is tried.
IBKR_RECONNECT_ENABLED            = True
# Maximum TOTAL connect attempts for one one-shot run (the first try PLUS
# retries). Bounded by design: the retry loop ALWAYS terminates after this many
# tries — it never retries forever. Must be >= 1 (clamped up to 1 if set lower).
IBKR_RECONNECT_MAX_ATTEMPTS       = 3
# Exponential-backoff base delay (seconds) between failed connect attempts:
#   delay(n) = min(MAX_DELAY, BASE_DELAY * 2**n)   for the n-th retry (0-indexed)
IBKR_RECONNECT_BASE_DELAY_SECONDS = 2.0
# Hard ceiling (seconds) on any single backoff delay, so the bounded retry can
# never sleep for an unreasonable stretch (worst-case total stays small).
IBKR_RECONNECT_MAX_DELAY_SECONDS  = 30.0
# Socket/request timeout (seconds) handed to ib_insync (ib.RequestTimeout and the
# connect() timeout), mirroring flatten_vti.py / check_positions.py. Keeps a dead
# TWS from hanging a one-shot run indefinitely. 0 disables the explicit timeout.
IBKR_REQUEST_TIMEOUT_SECONDS      = 30.0
