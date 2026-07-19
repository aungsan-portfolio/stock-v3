"""
correlation.py -- Pairwise correlation and portfolio overlap risk.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import pandas as pd

import config
from strategies.intraday_data import fetch_intraday

logger = logging.getLogger(__name__)

# Cache: { (symbol1, symbol2, interval, lookback): (correlation_value, timestamp) }
_CORRELATION_CACHE = {}


@dataclass
class CorrelationResult:
    symbol: str
    max_corr: float
    avg_corr: float
    penalty: float
    mode: str
    compared_symbols: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PortfolioPosition:
    symbol: str
    side: str
    exposure: float


class CorrelationAnalyzer:
    def __init__(self):
        self.ttl = getattr(config, "CORRELATION_CACHE_TTL_SECONDS", 300)
        self.timeframe = getattr(config, "CORRELATION_TIMEFRAME", "5m")
        self.lookback = getattr(config, "CORRELATION_LOOKBACK_BARS", 30)
        self.use_abs = getattr(config, "CORRELATION_USE_ABS", True)

    def get_returns(self, symbol: str, bridge=None) -> Optional[pd.Series]:
        """Fetch OHLC bars and compute returns."""
        try:
            # Dynamically calculate required days based on timeframe and lookback bars
            bars_per_day = 78 if self.timeframe == "5m" else 390
            lookback_days = max(2, (self.lookback // bars_per_day) + 3)
            
            df = fetch_intraday(symbol, interval=self.timeframe, lookback_days=lookback_days, bridge=bridge)
            if df.empty or len(df) < 10:
                return None
            # Return last N returns
            returns = df['close'].pct_change().dropna().tail(self.lookback)
            if len(returns) < 10:
                return None
            return returns
        except Exception as e:
            logger.warning(f"Failed to fetch returns for {symbol}: {e}")
            return None

    def get_returns_map(self, symbols: List[str], bridge=None) -> Dict[str, pd.Series]:
        """Fetch each symbol's returns once for a portfolio evaluation cycle."""
        returns = {}
        for symbol in dict.fromkeys(symbols):
            symbol_returns = self.get_returns(symbol, bridge=bridge)
            if symbol_returns is not None:
                returns[symbol] = symbol_returns
        return returns

    def pairwise_correlation(self, sym_a: str, sym_b: str, returns_a: pd.Series, returns_b: pd.Series) -> float:
        """Compute Pearson correlation of aligned return series."""
        df_merged = pd.concat([returns_a, returns_b], axis=1, join="inner").dropna()
        
        min_samples = getattr(config, "MIN_CORRELATION_SAMPLES", 10)
        if len(df_merged) < min_samples:
            logger.warning(
                f"Insufficient overlapping samples ({len(df_merged)} < {min_samples}) for {sym_a} and {sym_b}. "
                f"Defaulting to 0.0 correlation penalty."
            )
            return 0.0

        corr = df_merged.iloc[:, 0].corr(df_merged.iloc[:, 1])
        if pd.isna(corr):
            logger.warning(
                f"Correlation calculation returned NaN for {sym_a} and {sym_b}. "
                f"Defaulting to 0.0 correlation penalty."
            )
            return 0.0
        
        return abs(corr) if self.use_abs else corr

    def analyze(
        self,
        signal,
        precomputed_returns: Dict[str, pd.Series],
        bridge=None,
        positions: Optional[List[PortfolioPosition]] = None,
        candidate_exposure: Optional[float] = None,
    ) -> CorrelationResult:
        """Compare a single signal against precomputed returns of open positions."""
        symbol = signal.symbol
        compared_symbols = []
        correlations = []
        positions_by_symbol = {
            position.symbol: position
            for position in (positions or [])
        }
        candidate_direction = 1.0 if str(signal.side).upper() == "BUY" else -1.0
        
        sig_returns = precomputed_returns.get(symbol)
        
        if sig_returns is None or sig_returns.empty:
            return CorrelationResult(symbol, 0.0, 0.0, 1.0, "neutral", [])

        now = time.time()
        
        for open_sym, open_returns in precomputed_returns.items():
            if open_sym == symbol:
                continue
                
            is_ibkr = True if bridge and bridge.is_connected else False
            sorted_pair = tuple(sorted([symbol, open_sym]))
            cache_key = (*sorted_pair, self.timeframe, self.lookback, is_ibkr, self.use_abs)
            
            cached_val, cached_ts = _CORRELATION_CACHE.get(cache_key, (None, 0))
            if cached_val is not None and (now - cached_ts) < self.ttl:
                corr = cached_val
            else:
                corr = self.pairwise_correlation(symbol, open_sym, sig_returns, open_returns)
                self._prune_cache(now)
                _CORRELATION_CACHE[cache_key] = (corr, now)
                
            position = positions_by_symbol.get(
                open_sym,
                PortfolioPosition(open_sym, "BUY", candidate_exposure or 1.0),
            )
            position_direction = 1.0 if position.side.upper() == "BUY" else -1.0
            if candidate_exposure and candidate_exposure > 0 and position.exposure > 0:
                exposure_scale = min(candidate_exposure, position.exposure) / max(
                    candidate_exposure,
                    position.exposure,
                )
            else:
                exposure_scale = 1.0
            effective_corr = corr * candidate_direction * position_direction * exposure_scale
            correlations.append(effective_corr)
            compared_symbols.append(open_sym)
            
        if not correlations:
            return CorrelationResult(symbol, 0.0, 0.0, 1.0, "neutral", [])
            
        max_corr = max(0.0, max(correlations))
        avg_corr = sum(correlations) / len(correlations)
        
        penalty = self._compute_penalty(max_corr)
        mode = "penalty"
        
        if max_corr >= getattr(config, "CORRELATION_REJECT_THRESHOLD", 0.80):
            if getattr(config, "CORRELATION_CONSERVATIVE_REJECT", False):
                mode = "reject"
                
        return CorrelationResult(symbol, max_corr, avg_corr, penalty, mode, compared_symbols)

    def check_portfolio_impact(
        self,
        symbol: str,
        side: str,
        open_positions: List[PortfolioPosition],
        equity: float,
        bridge=None,
        precomputed_returns: Optional[Dict[str, pd.Series]] = None,
        candidate_exposure: Optional[float] = None,
    ) -> tuple:
        """Check if a new trade violates portfolio correlation limits."""
        if not open_positions:
            return True, 1.0, 0.0, []
            
        normalized_positions = [
            position
            if isinstance(position, PortfolioPosition)
            else PortfolioPosition(str(position), "BUY", candidate_exposure or equity)
            for position in open_positions
        ]
        position_symbols = [position.symbol for position in normalized_positions]
        symbols = list(dict.fromkeys([symbol, *position_symbols]))
        returns = (
            precomputed_returns
            if precomputed_returns is not None
            else self.get_returns_map(symbols, bridge=bridge)
        )
        portfolio_returns = {
            item: returns[item]
            for item in symbols
            if item in returns
        }
        missing_symbols = [item for item in symbols if item not in portfolio_returns]
        if missing_symbols:
            penalty = getattr(config, "CORRELATION_HARD_PENALTY", 0.50)
            allowed = not getattr(config, "CORRELATION_CONSERVATIVE_REJECT", False)
            return allowed, penalty, 1.0, missing_symbols
        result = self.analyze(
            type("PortfolioSignal", (), {"symbol": symbol, "side": side})(),
            portfolio_returns,
            bridge=bridge,
            positions=normalized_positions,
            candidate_exposure=candidate_exposure,
        )
        allowed = result.mode != "reject"
        return allowed, result.penalty, result.max_corr, result.compared_symbols

    def _prune_cache(self, now: float) -> None:
        expired = [
            key
            for key, (_, cached_at) in _CORRELATION_CACHE.items()
            if now - cached_at >= self.ttl
        ]
        for key in expired:
            _CORRELATION_CACHE.pop(key, None)

        max_entries = max(
            1,
            int(getattr(config, "CORRELATION_CACHE_MAX_ENTRIES", 5000)),
        )
        overflow = len(_CORRELATION_CACHE) - max_entries + 1
        if overflow > 0:
            oldest = sorted(
                _CORRELATION_CACHE,
                key=lambda key: _CORRELATION_CACHE[key][1],
            )[:overflow]
            for key in oldest:
                _CORRELATION_CACHE.pop(key, None)

    def _compute_penalty(self, max_corr: float) -> float:
        if max_corr < getattr(config, "CORRELATION_WARN_THRESHOLD", 0.60):
            if max_corr < 0.30:
                return 1.0
            return getattr(config, "CORRELATION_MILD_PENALTY", 0.90)
        elif max_corr < getattr(config, "CORRELATION_REJECT_THRESHOLD", 0.80):
            return getattr(config, "CORRELATION_STRONG_PENALTY", 0.75)
        else:
            return getattr(config, "CORRELATION_HARD_PENALTY", 0.50)
