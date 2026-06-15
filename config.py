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
