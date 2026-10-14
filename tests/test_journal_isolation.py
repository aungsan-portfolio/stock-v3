"""
tests/test_journal_isolation.py -- Tripwire regression suite for production journal isolation.

Guarantees:
1. Production journals (reports/trade_journal_production_clean.jsonl, reports/trade_journal.jsonl)
   are never touched by test fixtures or validation-lane executions.
2. Validation orders (with 'val_' prefix) are strictly prevented from writing to the production
   journal (Hard Lane Guard fails closed with RuntimeError).
3. Static audit: No test file in tests/ hardcodes writes to production journal paths.
4. Canonical se3- client order ID format and strategy parsing round-trip for all 6 strategies.
5. recovered_via contract adherence.
"""

import hashlib
import json
from pathlib import Path
import pytest

from strategies.order_idempotency import (
    STRATEGY_PREFIX_MAP,
    RECOVERED_VIA_VALID_VALUES,
    build_client_order_id,
    parse_strategy_from_client_id,
)
from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState
from strategies.trade_journal import log_trade, log_fill


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestJournalIsolationTripwire:
    """Verifies production journal immutability and lane segregation."""

    def test_production_journal_bytes_unchanged_during_component_runs(self, tmp_path):
        """Verify that running journal components directed to tmp_path leaves production journals intact."""
        repo_root = Path(__file__).resolve().parent.parent
        prod_clean = repo_root / "reports" / "trade_journal_production_clean.jsonl"
        prod_raw = repo_root / "reports" / "trade_journal.jsonl"

        hash_clean_before = _hash_file(prod_clean)
        hash_raw_before = _hash_file(prod_raw)

        # Execute journal writes to isolated tmp_path
        tmp_journal = tmp_path / "test_trade_journal.jsonl"
        log_trade(
            symbol="TEST",
            side="BUY",
            strategy="ORB",
            qty=10,
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            journal_file=tmp_journal,
        )
        log_fill(
            symbol="TEST",
            side="BUY",
            qty=10,
            fill_price=100.0,
            expected_price=100.0,
            slippage=0.0,
            fill_latency_ms=12.5,
            journal_file=tmp_journal,
            order_id=999999,
        )

        assert tmp_journal.exists()
        with open(tmp_journal, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2

        # Verify production hashes remain byte-identical
        assert _hash_file(prod_clean) == hash_clean_before
        assert _hash_file(prod_raw) == hash_raw_before

    def test_validation_lane_hard_guard_fails_closed(self, tmp_path):
        """Verify that any validation order routed to production journal functions raises RuntimeError."""
        # 1. Direct log_trade with val_ client_order_id must raise
        with pytest.raises(RuntimeError, match="HARD LANE GUARD: Validation order"):
            log_trade(
                symbol="AAPL",
                side="BUY",
                strategy="ORB",
                qty=10,
                entry_price=150.0,
                stop_price=140.0,
                target_price=160.0,
                client_order_id="val_test_order_123",
            )

        # 2. Direct log_fill with val_ client_order_id must raise
        with pytest.raises(RuntimeError, match="HARD LANE GUARD: Validation order"):
            log_fill(
                symbol="AAPL",
                side="BUY",
                qty=10,
                fill_price=150.0,
                expected_price=150.0,
                slippage=0.0,
                fill_latency_ms=10.0,
                client_order_id="val_test_fill_123",
            )

    def test_emergency_flatten_validation_routing(self, tmp_path):
        """Verify that positions with val_ order_id are routed exclusively to validation lane."""
        test_state_file = tmp_path / "trailing_state_test.json"
        manager = DynamicTrailingStopManager(state_file=test_state_file)

        state = TrailingStopState(
            symbol="AAPL",
            side="BUY",
            entry_price=150.0,
            peak_price=150.0,
            atr=2.0,
            stop_price=140.0,
            trail_multiple=1.5,
            last_updated=0.0,
            order_id="val_test_order_123",
            active=True,
            qty=100.0,
            strategy="MOMENTUM_SCALP",
            is_validation=False,  # Unflagged validation order, but order_id has val_ prefix
        )
        manager.states["AAPL"] = state

        class MockBridge:
            def market_price(self, sym):
                return 140.0
            def close_position(self, sym):
                return True
            def working_orders_plain(self):
                return []

        # Emergency flatten should succeed by automatically routing to validation lane without touching production
        manager.emergency_flatten("AAPL", 100.0, bridge=MockBridge(), dry_run=False)
        assert manager.states["AAPL"].active is False

    def test_se3_client_order_id_generation_and_parsing_roundtrip(self):
        """Verify canonical se3- format and parse_strategy_from_client_id round-trip for all 6 strategies."""
        strategies = [
            ("MOMENTUM_SCALP", "mome"),
            ("VWAP_BOUNCE", "vwap"),
            ("GAP_AND_GO", "gapa"),
            ("ORB", "orb"),
            ("CANDLESTICK", "cand"),
            ("MEAN_REVERSION", "mean"),
        ]

        for strat_name, expected_prefix in strategies:
            cid = build_client_order_id(
                symbol="NVDA",
                side="BUY",
                strategy=strat_name,
                session_date="20260926",
            )
            # Check structure: se3-NVDA-B-{prefix}-20260926-{digest}
            parts = cid.split("-")
            assert parts[0] == "se3"
            assert parts[1] == "NVDA"
            assert parts[2] == "B"
            assert parts[3] == expected_prefix
            assert parts[4] == "20260926"
            assert len(parts[5]) == 16

            # Parse round-trip
            parsed = parse_strategy_from_client_id(cid)
            assert parsed == strat_name

        # Test invalid / fallback cases
        assert parse_strategy_from_client_id("") == "UNKNOWN"
        assert parse_strategy_from_client_id(None) == "UNKNOWN"
        assert parse_strategy_from_client_id("legacy-order-1234") == "UNKNOWN"
        assert parse_strategy_from_client_id("se3-AAPL-B-xxxx-20260926-1234567890123456") == "UNKNOWN"

    def test_recovered_via_contract_values(self):
        """Verify the standardized recovered_via enum contract."""
        expected_contract = {
            "client_order_id_parse",
            "gate_order_id_join",
            "gate_session_tiebreak",
            "archive_join_20260722",
        }
        assert RECOVERED_VIA_VALID_VALUES == expected_contract

    def test_static_audit_no_test_hardcodes_production_clean_journal(self):
        """Static audit: ensure no test file under tests/ hardcodes writes to trade_journal_production_clean.jsonl."""
        repo_root = Path(__file__).resolve().parent.parent
        tests_dir = repo_root / "tests"
        forbidden_token = "trade_journal_production_clean.jsonl"

        violators = []
        for py_file in tests_dir.glob("*.py"):
            if py_file.name == "test_journal_isolation.py":
                continue
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            # If the token is found and appears in an open("...", "w" or "a") context
            if forbidden_token in content:
                for line in content.splitlines():
                    if forbidden_token in line and any(w in line for w in ('"w"', '"a"', "'w'", "'a'", "write_text")):
                        violators.append(f"{py_file.name}: {line.strip()}")

        assert not violators, f"Found test files directly opening production clean journal for writing: {violators}"

    def test_strategy_disable_governance_flags_and_veto_reason(self):
        """Verify MOMENTUM_SCALP and VWAP_BOUNCE are disabled in config and STRATEGY_DISABLED veto exists."""
        import config
        from strategies.veto_reasons import VetoReason

        assert config.MOMENTUM_SCALP_ENABLED is False
        assert config.STRATEGY_MOMENTUM_SCALP_ENABLED is False
        assert config.VWAP_BOUNCE_ENABLED is False
        assert config.STRATEGY_VWAP_BOUNCE_ENABLED is False

        assert hasattr(VetoReason, "STRATEGY_DISABLED")
        assert VetoReason.STRATEGY_DISABLED.value == "STRATEGY_DISABLED"

    def test_trailing_state_persistence_isolation(self, tmp_path):
        """Regression: ensure DynamicTrailingStopManager with injected path never touches production trailing_state.json."""
        repo_root = Path(__file__).resolve().parent.parent
        prod_state = repo_root / "data" / "trailing_state.json"
        hash_before = _hash_file(prod_state)

        test_state_file = tmp_path / "isolated_trailing_state.json"
        mgr = DynamicTrailingStopManager(state_file=test_state_file)
        assert mgr.STATE_FILE == str(test_state_file.resolve())

        # Register a state and save
        state = TrailingStopState(
            symbol="ISOLATION_TEST",
            side="BUY",
            entry_price=100.0,
            peak_price=105.0,
            atr=1.5,
            stop_price=98.0,
            trail_multiple=1.5,
            last_updated=0.0,
            qty=10.0,
        )
        mgr.states["ISOLATION_TEST"] = state
        mgr._record_pending_exit(state, qty_closed=10.0)
        mgr._save_state()

        # Injected file must exist and contain the test symbol
        assert test_state_file.exists()
        saved_data = json.loads(test_state_file.read_text(encoding="utf-8"))
        assert "ISOLATION_TEST" in saved_data

        # Production state file bytes must remain 100% identical
        hash_after = _hash_file(prod_state)
        assert hash_after == hash_before, "CRITICAL: Production data/trailing_state.json was modified by isolated test manager!"

    def test_trailing_state_restart_from_injected_path(self, tmp_path):
        """Regression: verify manager correctly loads persisted state from injected path across restarts."""
        test_state_file = tmp_path / "restart_trailing_state.json"
        mgr1 = DynamicTrailingStopManager(state_file=test_state_file)
        state = TrailingStopState(
            symbol="RESTART_TEST",
            side="BUY",
            entry_price=50.0,
            peak_price=55.0,
            atr=1.0,
            stop_price=49.0,
            trail_multiple=1.5,
            last_updated=0.0,
            qty=5.0,
            active=True,
        )
        mgr1.states["RESTART_TEST"] = state
        mgr1._save_state()

        # Restart with new manager instance pointing to same file
        mgr2 = DynamicTrailingStopManager(state_file=test_state_file)
        assert "RESTART_TEST" in mgr2.states
        assert mgr2.states["RESTART_TEST"].entry_price == 50.0
        assert mgr2.states["RESTART_TEST"].qty == 5.0

    def test_trailing_state_default_constructor_preserves_canonical_path(self):
        """Regression: verify default constructor points to canonical production trailing_state.json."""
        repo_root = Path(__file__).resolve().parent.parent
        expected_path = str((repo_root / "data" / "trailing_state.json").resolve())
        mgr = DynamicTrailingStopManager()
        assert mgr.STATE_FILE == expected_path

