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
BUY_THRESHOLD       = 0.60
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
MAX_POSITION_PCT    = 0.20
STOP_LOSS_PCT       = 0.05
TAKE_PROFIT_PCT     = 0.15
MAX_OPEN_POSITIONS  = 5

# ── Production Safety Guards ─────────────────────
# Default is long-only. SELL closes existing long positions; it does not open shorts.
ALLOW_SHORT          = False
MIN_TRADE_CASH       = 100.0
LIMIT_ORDER_OFFSET_PCT = 0.001

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
IBKR_CLIENT_ID      = 1
PAPER_CAPITAL       = 100_000.0

# ── Logging ──────────────────────────────────────
LOG_FILE            = LOG_DIR / "stock_engine.log"
LOG_MAX_BYTES       = 10 * 1024 * 1024
LOG_BACKUP_COUNT    = 5
