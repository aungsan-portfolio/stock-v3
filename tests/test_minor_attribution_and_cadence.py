"""tests/test_minor_attribution_and_cadence.py -- Targeted tests for Lane 7 (Strategy attribution & Cadence sync label)."""

import json
import os
import tempfile
import time
from datetime import datetime, timezone
import pytest
from unittest.mock import MagicMock, patch

from strategies.trade_journal import log_trade, read_journal
from strategies.order_registry import register_order_expected_price


def test_strategy_attribution_from_client_order_id_and_parent():
    """Verify strategy attribution is resolved at write time without falling back to UNKNOWN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_path = os.path.join(tmpdir, "trade_journal.jsonl")

        # 1. Parent order registered in order_registry
        register_order_expected_price(
            order_id="parent-ord-1",
            expected_price=150.0,
            side="BUY",
            order_type="ENTRY",
            submit_ts=time.time(),
            symbol="AAPL",
            strategy="MOMENTUM_SCALP",
            client_order_id="dt_MOMENTUM_SCALP_parent1",
        )

        # 2. Child stop leg registered with parent_order_id reference
        register_order_expected_price(
            order_id="stop-leg-1",
            expected_price=148.0,
            side="SELL",
            order_type="STOP_LOSS",
            submit_ts=time.time(),
            symbol="AAPL",
            parent_order_id="parent-ord-1",
        )

        from pathlib import Path
        from strategies.trade_journal import log_fill
        j_file = Path(journal_path)

        # Log child exit fill with strategy=None
        log_fill(
            order_id="stop-leg-1",
            symbol="AAPL",
            side="SELL",
            qty=10,
            fill_price=147.95,
            expected_price=148.0,
            slippage=-0.05,
            fill_latency_ms=12.0,
            strategy=None,
            journal_file=j_file,
        )

        records = read_journal(j_file)
        assert len(records) == 1
        rec = records[0]
        # Must resolve to MOMENTUM_SCALP from parent order, not UNKNOWN
        assert rec.get("strategy") == "MOMENTUM_SCALP"


def test_cadence_sync_opening_buffer_vs_boot_label(caplog):
    """Verify startup cadence sync labels Opening buffer cleared mid-bar vs Engine booted mid-bar."""
    import logging
    import strategies.orchestrator as orch
    from strategies.orchestrator import EvaluationResult
    from strategies.regime_engine import MarketRegimeResult, MODE_RISK_ON, TEXTURE_TRENDING

    orch._last_evaluated_5m_bar_str = None
    orch._skipped_bar_ids.clear()

    bridge_mock = MagicMock()
    bridge_mock.get_positions.return_value = []
    bridge_mock.get_open_orders.return_value = []
    bridge_mock.get_net_liquidation.return_value = 100000.0

    regime = MarketRegimeResult(
        mode=MODE_RISK_ON,
        score=0,
        allow_new_longs=True,
        requires_opening_buffer=False,
        position_scale=1.0,
        confidence_boost=0.0,
        market_texture=TEXTURE_TRENDING,
    )

    # Simulate mid-bar tick at 14:02:30 UTC (150s elapsed > grace: 45s)
    now_utc = datetime(2026, 10, 1, 14, 2, 30, tzinfo=timezone.utc)
    now_et = datetime(2026, 10, 1, 10, 2, 30)

    with caplog.at_level(logging.INFO):
        # Case A: Engine booted 300s ago (> 180s) and opening buffer cleared
        with patch("strategies.orchestrator.datetime") as mock_dt, \
             patch("strategies.regime_engine.evaluate_market_regime", return_value=regime), \
             patch("strategies.session.is_market_open", return_value=True), \
             patch("strategies.session.is_past_opening_buffer", return_value=True), \
             patch("strategies.session.now_eastern", return_value=now_et), \
             patch.object(orch, "_engine_boot_ts", time.time() - 300.0):

            mock_dt.now.return_value = now_utc
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            orch.evaluate_and_execute(
                watchlist=["AAPL"],
                bridge=bridge_mock,
                live_paper=False,
            )

    sync_logs = [r.message for r in caplog.records if "[STARTUP CADENCE SYNC]" in r.message]
    assert len(sync_logs) == 1
    assert "Opening buffer cleared mid-bar" in sync_logs[0]
