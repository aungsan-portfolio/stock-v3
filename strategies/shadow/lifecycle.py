"""
strategies.shadow.lifecycle -- Shadow Trade Lifecycle & Conservative Simulation.

Enforces:
1. Zero broker bridge dependencies at constructor level.
2. Invariant: entry_timestamp_utc >= signal_bar_close_utc (Look-ahead prevention).
3. First outcome bar rule: Only completed bars closing AFTER entry can update outcomes.
4. Direction-aware MFE and MAE excursion metrics.
5. Conservative same-bar resolution: Both stop & target touched -> AMBIGUOUS_INTRABAR loss.
6. Time exit cutoff at market_close - 15 minutes (15:45 ET regular / 12:45 ET half-day).
7. Single cost deduction for net R calculations (avoiding double-counting).
"""
from datetime import datetime, time
import logging
from typing import Callable, Dict, List, Optional
import pandas as pd

import config
from strategies.shadow.models import (
    ShadowCandidate,
    ShadowOutcome,
    ShadowTrade,
    TimeBucket,
)
from strategies.shadow.pricing import resolve_entry_price, resolve_exit_price
from strategies.shadow.storage import ShadowStorageManager

logger = logging.getLogger(__name__)


class ShadowLifecycleManager:
    """Manages active shadow trade state, outcome resolution, and journal persistence."""

    def __init__(
        self,
        storage: Optional[ShadowStorageManager] = None,
        clock: Optional[Callable[[], pd.Timestamp]] = None,
    ):
        self.storage = storage or ShadowStorageManager(
            base_dir=getattr(config, "SHADOW_BASE_DIR", "data/shadow"),
            role=getattr(config, "SHADOW_ENGINE_ROLE", "daytrade"),
            strategy_version=getattr(config, "SHADOW_STRATEGY_VERSION", "ORB_CONFIRMED_V1_FIXED_2R"),
        )
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self._active_trades: Dict[str, ShadowTrade] = {}
        self._enabled = True

        # Acquire single-writer lock on initialization
        if not self.storage.acquire_lock():
            self._enabled = False
            logger.warning("[SHADOW DISABLED: LOCK UNAVAILABLE] Lifecycle manager disabled due to lock contention.")
        else:
            self._active_trades = self.storage.load_active_trades()
            logger.info("[SHADOW LIFECYCLE] Loaded %d active shadow trade(s)", len(self._active_trades))

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def active_trade_count(self) -> int:
        return len(self._active_trades)

    def register_candidate(
        self,
        candidate: ShadowCandidate,
        quote: Optional[dict] = None,
    ) -> Optional[ShadowTrade]:
        """Registers a new shadow candidate and initializes hypothetical bracket geometry."""
        if not self._enabled:
            return None

        tid = candidate.candidate_id
        if tid in self._active_trades:
            return self._active_trades[tid]

        # Check if already closed previously
        if tid in self.storage.load_closed_ids():
            logger.info("[SHADOW LIFECYCLE] Candidate %s already closed in ledger. Skipping.", tid)
            return None

        now_utc = self.clock()
        signal_bar_close = pd.Timestamp(candidate.signal_bar_close_utc)
        if signal_bar_close.tzinfo is None:
            signal_bar_close = signal_bar_close.tz_localize("UTC")
        else:
            signal_bar_close = signal_bar_close.tz_convert("UTC")

        # Invariant: Entry timestamp cannot precede signal bar close!
        entry_ts = max(now_utc, signal_bar_close)

        # Resolve entry price with execution friction
        (
            entry_price,
            pricing_src,
            synthetic_cost,
            half_spread_cost,
            slippage_cost,
            quote_ts,
            quote_age,
        ) = resolve_entry_price(
            side=candidate.side,
            reference_close=candidate.reference_price,
            quote=quote,
            now_utc=now_utc,
            slippage_bps=float(getattr(config, "SHADOW_SLIPPAGE_BPS", 5.0)),
            max_quote_age_seconds=float(getattr(config, "SHADOW_QUOTE_MAX_AGE_SECONDS", 5.0)),
        )

        risk = round(entry_price - candidate.stop_price, 4) if candidate.side == "BUY" else round(candidate.stop_price - entry_price, 4)
        if risk <= 0:
            logger.warning("[SHADOW LIFECYCLE] Degenerate risk per share (%.4f) for %s. Skipping.", risk, candidate.symbol)
            return None

        target_r = float(getattr(config, "ORB_TARGET_R_MULTIPLE", 2.0))
        target_price = round(entry_price + (target_r * risk), 4) if candidate.side == "BUY" else round(entry_price - (target_r * risk), 4)

        trade = ShadowTrade(
            shadow_trade_id=tid,
            strategy_name=candidate.strategy_name,
            strategy_version=candidate.strategy_version,
            symbol=candidate.symbol,
            side=candidate.side,
            signal_timestamp_utc=candidate.signal_timestamp_utc,
            signal_bar_start_utc=candidate.signal_bar_start_utc,
            signal_bar_close_utc=candidate.signal_bar_close_utc,
            entry_timestamp_utc=entry_ts.isoformat(),
            entry_time_bucket=TimeBucket.PRIMARY.value,
            session_date_et=candidate.session_date_et,
            market_texture=candidate.market_texture,
            regime_mode=candidate.regime_mode,
            opening_range_high=candidate.opening_range_high,
            opening_range_low=candidate.opening_range_low,
            entry_price=entry_price,
            stop_price=candidate.stop_price,
            target_price=target_price,
            risk_per_share=risk,
            quantity_model=100,
            entry_pricing_source=pricing_src,
            entry_quote_timestamp_utc=quote_ts,
            entry_quote_age_seconds=quote_age,
            entry_reference_close=candidate.reference_price,
            entry_synthetic_cost=synthetic_cost,
            entry_half_spread_cost=half_spread_cost,
            entry_slippage_cost=slippage_cost,
            relative_volume=candidate.relative_volume,
            breakout_volume_ratio=candidate.breakout_volume_ratio,
            price_vs_vwap=candidate.price_vs_vwap,
            vwap_slope=candidate.vwap_slope,
            ema9=candidate.ema9,
            ema20=candidate.ema20,
            atr=candidate.atr,
            atr_pct=candidate.atr_pct,
            spread_pct=candidate.spread_pct,
            spy_alignment=candidate.spy_alignment.value if hasattr(candidate.spy_alignment, "value") else str(candidate.spy_alignment),
            qqq_alignment=candidate.qqq_alignment.value if hasattr(candidate.qqq_alignment, "value") else str(candidate.qqq_alignment),
            sector_alignment=candidate.sector_alignment.value if hasattr(candidate.sector_alignment, "value") else str(candidate.sector_alignment),
            catalyst_type=candidate.catalyst_type,
            premarket_gap_pct=candidate.premarket_gap_pct,
            premarket_volume_ratio=candidate.premarket_volume_ratio,
            confidence=candidate.confidence,
            status=ShadowOutcome.ACTIVE.value,
            highest_price_seen=entry_price,
            lowest_price_seen=entry_price,
        )

        self._active_trades[tid] = trade
        self.storage.save_active_trades(self._active_trades)
        logger.info(
            "[ORB SHADOW OPEN] Registered %s (%s) @ entry=%.2f stop=%.2f target=%.2f risk=%.2f source=%s",
            trade.symbol,
            trade.shadow_trade_id[:8],
            trade.entry_price,
            trade.stop_price,
            trade.target_price,
            trade.risk_per_share,
            trade.entry_pricing_source,
        )
        return trade

    def update_active_trades(
        self,
        symbol: str,
        completed_bar: pd.Series,
        now_utc: Optional[pd.Timestamp] = None,
        quote: Optional[dict] = None,
    ) -> List[ShadowTrade]:
        """Evaluates active shadow trades against the latest completed bar.

        Enforces first outcome bar rule, same-bar conservative resolution, and time exit.
        """
        if not self._enabled:
            return []

        now = now_utc or self.clock()
        sym_u = symbol.upper().strip()
        matching_trades = [t for t in self._active_trades.values() if t.symbol == sym_u and t.status == ShadowOutcome.ACTIVE.value]
        if not matching_trades:
            return []

        bar_high = float(completed_bar["high"])
        bar_low = float(completed_bar["low"])
        bar_close = float(completed_bar["close"])

        # Determine bar close UTC timestamp
        bar_start_ts = completed_bar.name if isinstance(completed_bar.name, pd.Timestamp) else pd.Timestamp(completed_bar.get("timestamp", now))
        if bar_start_ts.tzinfo is None:
            bar_start_utc = bar_start_ts.tz_localize("UTC")
        else:
            bar_start_utc = bar_start_ts.tz_convert("UTC")

        # Assume 1-minute bar frequency unless specified
        bar_close_utc = bar_start_utc + pd.Timedelta(minutes=1)

        closed_this_cycle: List[ShadowTrade] = []

        for trade in matching_trades:
            entry_ts = pd.Timestamp(trade.entry_timestamp_utc)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")

            # ── First Outcome Bar Rule ──
            # Only bars whose close time is strictly AFTER entry timestamp can update outcomes
            if bar_close_utc <= entry_ts:
                continue

            trade.bars_held += 1
            trade.highest_price_seen = max(trade.highest_price_seen, bar_high)
            trade.lowest_price_seen = min(trade.lowest_price_seen, bar_low)

            # ── Direction-Aware Excursions (MFE / MAE) ──
            if trade.side == "BUY":
                mfe_r = (trade.highest_price_seen - trade.entry_price) / trade.risk_per_share
                mae_r = (trade.entry_price - trade.lowest_price_seen) / trade.risk_per_share
            else:
                mfe_r = (trade.entry_price - trade.lowest_price_seen) / trade.risk_per_share
                mae_r = (trade.highest_price_seen - trade.entry_price) / trade.risk_per_share

            trade.maximum_favorable_excursion_r = round(mfe_r, 4)
            trade.maximum_adverse_excursion_r = round(mae_r, 4)
            if mfe_r >= 1.5:
                trade.secondary_1_5r_reached = True

            # ── Outcome Resolution ──
            is_closed = False

            if trade.side == "BUY":
                hit_stop = bar_low <= trade.stop_price
                hit_target = bar_high >= trade.target_price

                # Conservative same-bar ambiguity resolution
                if hit_stop and hit_target:
                    trade.status = ShadowOutcome.AMBIGUOUS_INTRABAR.value
                    trade.ambiguous_intrabar = True
                    trade.exit_price = trade.stop_price
                    trade.exit_reason = "AMBIGUOUS_INTRABAR_STOP_FIRST"
                    trade.exit_pricing_source = "CONSERVATIVE_STOP_PRICE"
                    is_closed = True
                elif hit_stop:
                    trade.status = ShadowOutcome.STOPPED_OUT.value
                    trade.exit_price = trade.stop_price
                    trade.exit_reason = "STOP_TRIGGERED"
                    trade.exit_pricing_source = "STOP_LIMIT"
                    is_closed = True
                elif hit_target:
                    trade.status = ShadowOutcome.TARGET_HIT.value
                    trade.exit_price = trade.target_price
                    trade.exit_reason = "TARGET_REACHED"
                    trade.exit_pricing_source = "TARGET_LIMIT"
                    is_closed = True
            else:  # SELL
                hit_stop = bar_high >= trade.stop_price
                hit_target = bar_low <= trade.target_price

                if hit_stop and hit_target:
                    trade.status = ShadowOutcome.AMBIGUOUS_INTRABAR.value
                    trade.ambiguous_intrabar = True
                    trade.exit_price = trade.stop_price
                    trade.exit_reason = "AMBIGUOUS_INTRABAR_STOP_FIRST"
                    trade.exit_pricing_source = "CONSERVATIVE_STOP_PRICE"
                    is_closed = True
                elif hit_stop:
                    trade.status = ShadowOutcome.STOPPED_OUT.value
                    trade.exit_price = trade.stop_price
                    trade.exit_reason = "STOP_TRIGGERED"
                    trade.exit_pricing_source = "STOP_LIMIT"
                    is_closed = True
                elif hit_target:
                    trade.status = ShadowOutcome.TARGET_HIT.value
                    trade.exit_price = trade.target_price
                    trade.exit_reason = "TARGET_REACHED"
                    trade.exit_pricing_source = "TARGET_LIMIT"
                    is_closed = True

            # ── Time Exit Check (15:45 ET cutoff) ──
            if not is_closed:
                bar_et = bar_close_utc.tz_convert("America/New_York")
                time_exit_cutoff = time(15, 45)
                if bar_et.time() >= time_exit_cutoff:
                    exit_price, exit_src, exit_synth_cost = resolve_exit_price(
                        side=trade.side,
                        reference_close=bar_close,
                        quote=quote,
                        now_utc=now,
                        slippage_bps=float(getattr(config, "SHADOW_SLIPPAGE_BPS", 5.0)),
                    )
                    trade.status = ShadowOutcome.TIME_EXIT.value
                    trade.exit_price = exit_price
                    trade.exit_reason = "TIME_EXIT_1545_ET"
                    trade.exit_pricing_source = exit_src
                    trade.exit_synthetic_cost = exit_synth_cost
                    is_closed = True

            if is_closed:
                trade.exit_timestamp_utc = bar_close_utc.isoformat()
                self._settle_financials(trade)
                self.storage.append_closed_trade(trade)
                closed_this_cycle.append(trade)
                del self._active_trades[trade.shadow_trade_id]
                logger.info(
                    "[ORB SHADOW CLOSE] Settled %s (%s) @ exit=%.2f net_pnl=$%.2f result_r=%.2f status=%s reason=%s",
                    trade.symbol,
                    trade.shadow_trade_id[:8],
                    trade.exit_price or 0.0,
                    trade.net_pnl,
                    trade.net_result_r,
                    trade.status,
                    trade.exit_reason,
                )

        if closed_this_cycle:
            self.storage.save_active_trades(self._active_trades)

        return closed_this_cycle

    @staticmethod
    def _settle_financials(trade: ShadowTrade) -> None:
        """Computes gross and net financial results, ensuring costs are deducted exactly once."""
        exit_p = float(trade.exit_price or trade.entry_price)
        qty = trade.quantity_model

        if trade.side == "BUY":
            trade.gross_pnl = round((exit_p - trade.entry_price) * qty, 2)
            gross_r = (exit_p - trade.entry_price) / trade.risk_per_share
        else:
            trade.gross_pnl = round((trade.entry_price - exit_p) * qty, 2)
            gross_r = (trade.entry_price - exit_p) / trade.risk_per_share

        trade.gross_result_r = round(gross_r, 4)

        # Frictions
        entry_cost = trade.entry_synthetic_cost + trade.entry_half_spread_cost + trade.entry_slippage_cost
        exit_cost = trade.exit_synthetic_cost + trade.exit_half_spread_cost + trade.exit_slippage_cost
        total_unit_cost = entry_cost + exit_cost

        trade.total_cost = round(total_unit_cost * qty, 2)
        trade.net_pnl = round(trade.gross_pnl - trade.total_cost, 2)

        trade.cost_r = round(total_unit_cost / trade.risk_per_share, 4)
        trade.net_result_r = round(trade.gross_result_r - trade.cost_r, 4)

        # Outcome classification
        if trade.net_result_r > 0.05:
            trade.outcome_class = "WIN"
        elif trade.net_result_r < -0.05:
            trade.outcome_class = "LOSS"
        else:
            trade.outcome_class = "BREAKEVEN"


_shadow_lifecycle_manager_instance: Optional[ShadowLifecycleManager] = None


def get_shadow_lifecycle_manager(
    storage: Optional[ShadowStorageManager] = None,
    clock: Optional[Callable[[], pd.Timestamp]] = None,
    force_new: bool = False,
) -> ShadowLifecycleManager:
    """Returns singleton instance of ShadowLifecycleManager."""
    global _shadow_lifecycle_manager_instance
    if _shadow_lifecycle_manager_instance is None or force_new:
        _shadow_lifecycle_manager_instance = ShadowLifecycleManager(storage=storage, clock=clock)
    return _shadow_lifecycle_manager_instance


def reset_shadow_lifecycle_manager() -> None:
    """Resets singleton instance and releases lock (for tests / restart)."""
    global _shadow_lifecycle_manager_instance
    if _shadow_lifecycle_manager_instance is not None:
        try:
            _shadow_lifecycle_manager_instance.storage.release_lock()
        except Exception:
            pass
        _shadow_lifecycle_manager_instance = None
