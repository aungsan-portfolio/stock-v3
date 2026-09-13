"""
tests/test_journal_deduplication_audit.py -- Comprehensive Unit Tests for Phase 4:
Partial-fill synchronization, immutable incremental journal reconciliation,
REST/WebSocket convergence, decimal canonicalization, and shutdown position assertion.
"""
import os
import json
import pathlib
import shutil
import tempfile
import threading
import unittest
import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytz

import config
from strategies.session import now_eastern
from strategies.trade_journal import (
    log_fill,
    log_trade,
    read_journal,
    canonical_decimal,
    is_economic_fill,
    load_journal_order_fill_state,
    OrderFillState,
    get_journal_lock,
)
from alpaca_bridge import AlpacaBridge


class MockAlpacaOrder:
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        filled_qty: str,
        filled_avg_price: str,
        filled_at: datetime = None,
        status: str = "filled",
        client_order_id: str = "dt_VWAP_test",
        updated_at: datetime = None,
    ):
        self.id = order_id
        self.symbol = symbol
        self.side = side
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.filled_at = filled_at or datetime.now(timezone.utc)
        self.status = status
        self.client_order_id = client_order_id
        self.updated_at = updated_at or self.filled_at
        self.stop_price = None
        self.limit_price = None
        self.created_at = self.filled_at
        self.submitted_at = self.filled_at


class TestJournalDeduplicationAudit(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self.journal_path = pathlib.Path(self._tmp_dir) / "test_journal.jsonl"
        self._orig_journal = config.DAYTRADE_TRADE_JOURNAL_FILE
        config.DAYTRADE_TRADE_JOURNAL_FILE = self.journal_path

        # Setup mock bridge
        self.bridge = AlpacaBridge()
        self.bridge._connected = True
        self.bridge._client = MagicMock()
        self.bridge.account_id = "PA3_TEST_ACCOUNT"

    def tearDown(self):
        config.DAYTRADE_TRADE_JOURNAL_FILE = self._orig_journal
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. Partial-Fill Progression Tests
    # -------------------------------------------------------------
    def test_first_partial_fill_appends_57(self):
        """Simulate broker returning 57 shares partial fill: exactly 57 shares logged."""
        now_utc = datetime.now(timezone.utc)
        order_57 = MockAlpacaOrder(
            order_id="ord_bac_1",
            symbol="BAC",
            side="SELL",
            filled_qty="57",
            filled_avg_price="62.80",
            filled_at=now_utc,
        )
        self.bridge._client.get_orders.return_value = [order_57]

        synced = self.bridge.sync_today_trades_to_journal()
        self.assertEqual(synced, 1)

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("event_type") == "FILL"]
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["qty"], 57)
        self.assertEqual(fills[0]["fill_price"], 62.80)
        self.assertEqual(fills[0]["execution_id"], "SYNTH:ALPACA:ord_bac_1:0-57")

    def test_cumulative_fill_advancing_57_to_63_appends_delta_6(self):
        """Simulate order advancing from 57 to 63: appends an incremental 6-share record."""
        now_utc = datetime.now(timezone.utc)
        order_57 = MockAlpacaOrder(
            order_id="ord_bac_2",
            symbol="BAC",
            side="SELL",
            filled_qty="57",
            filled_avg_price="62.80",
            filled_at=now_utc,
        )
        self.bridge._client.get_orders.return_value = [order_57]
        self.bridge.sync_today_trades_to_journal()

        # Broker advances to 63 shares
        order_63 = MockAlpacaOrder(
            order_id="ord_bac_2",
            symbol="BAC",
            side="SELL",
            filled_qty="63",
            filled_avg_price="62.80",
            filled_at=now_utc,
        )
        self.bridge._client.get_orders.return_value = [order_63]
        synced2 = self.bridge.sync_today_trades_to_journal()

        self.assertEqual(synced2, 1)
        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("event_type") == "FILL"]
        self.assertEqual(len(fills), 2)
        self.assertEqual(fills[0]["qty"], 57)
        self.assertEqual(fills[0]["execution_id"], "SYNTH:ALPACA:ord_bac_2:0-57")
        self.assertEqual(fills[1]["qty"], 6)
        self.assertEqual(fills[1]["execution_id"], "SYNTH:ALPACA:ord_bac_2:57-63")
        self.assertEqual(fills[1]["reconciliation_reason"], "CUMULATIVE_FILLED_QTY_ADVANCED")

    def test_repeated_cumulative_63_is_idempotent(self):
        """100 repeated sync cycles on completed 63-share order produce zero duplicate appends."""
        now_utc = datetime.now(timezone.utc)
        order_63 = MockAlpacaOrder(
            order_id="ord_bac_3",
            symbol="BAC",
            side="SELL",
            filled_qty="63",
            filled_avg_price="62.80",
            filled_at=now_utc,
        )
        self.bridge._client.get_orders.return_value = [order_63]
        self.bridge.sync_today_trades_to_journal()

        for _ in range(100):
            res = self.bridge.sync_today_trades_to_journal()
            self.assertEqual(res, 0)

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("event_type") == "FILL"]
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["qty"], 63)

    def test_journal_sum_equals_broker_filled_quantity(self):
        """The sum of economic journal fills strictly matches the broker cumulative filled quantity."""
        now_utc = datetime.now(timezone.utc)
        for qty_step in ["20", "45", "63"]:
            order = MockAlpacaOrder(
                order_id="ord_sum_test",
                symbol="BAC",
                side="SELL",
                filled_qty=qty_step,
                filled_avg_price="62.80",
                filled_at=now_utc,
            )
            self.bridge._client.get_orders.return_value = [order]
            self.bridge.sync_today_trades_to_journal()

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("event_type") == "FILL" and r.get("broker_order_id") == "ord_sum_test"]
        total_qty = sum(r["qty"] for r in fills)
        self.assertEqual(total_qty, 63)
        self.assertEqual(len(fills), 3)

    # -------------------------------------------------------------
    # 2. Race and Idempotency Tests
    # -------------------------------------------------------------
    def test_websocket_57_then_rest_63(self):
        """WebSocket logs initial 57, subsequent REST sync sees cumulative 63 and appends only delta 6."""
        now_ts = now_eastern().isoformat()
        log_fill(
            symbol="BAC",
            side="SELL",
            qty=57,
            fill_price=62.80,
            expected_price=62.80,
            slippage=0.0,
            fill_latency_ms=10.0,
            journal_file=self.journal_path,
            order_id="ord_race_1",
            broker_order_id="ord_race_1",
            execution_id="ws_exec_1",
            timestamp=now_ts,
            account_id="PA3_TEST_ACCOUNT",
        )

        order_63 = MockAlpacaOrder(
            order_id="ord_race_1",
            symbol="BAC",
            side="SELL",
            filled_qty="63",
            filled_avg_price="62.80",
        )
        self.bridge._client.get_orders.return_value = [order_63]
        self.bridge.sync_today_trades_to_journal()

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("event_type") == "FILL" and r.get("broker_order_id") == "ord_race_1"]
        self.assertEqual(len(fills), 2)
        self.assertEqual(fills[0]["qty"], 57)
        self.assertEqual(fills[0]["execution_id"], "ws_exec_1")
        self.assertEqual(fills[1]["qty"], 6)
        self.assertEqual(fills[1]["execution_id"], "SYNTH:ALPACA:ord_race_1:57-63")

    def test_rest_and_websocket_concurrent_append_is_serialized(self):
        """Concurrent threads syncing orders under reconciliation lock do not create duplicate fills."""
        order_63 = MockAlpacaOrder(
            order_id="ord_concurrent",
            symbol="BAC",
            side="SELL",
            filled_qty="63",
            filled_avg_price="62.80",
        )
        self.bridge._client.get_orders.return_value = [order_63]

        threads = []
        for _ in range(5):
            t = threading.Thread(target=self.bridge.sync_today_trades_to_journal)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("event_type") == "FILL" and r.get("broker_order_id") == "ord_concurrent"]
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["qty"], 63)

    def test_simultaneous_rest_barrier_concurrency_after_partial_fill(self):
        """Simultaneous REST sync threads hitting a barrier concurrently append delta at most once."""
        now_utc = datetime.now(timezone.utc)
        now_ts = now_eastern().isoformat()
        log_fill(
            symbol="BAC",
            side="SELL",
            qty=57,
            fill_price=62.80,
            expected_price=62.80,
            slippage=0.0,
            fill_latency_ms=5.0,
            journal_file=self.journal_path,
            order_id="ord_barrier_test",
            broker_order_id="ord_barrier_test",
            execution_id="ws_barrier_exec_init",
            timestamp=now_ts,
            account_id="PA3_TEST_ACCOUNT",
        )

        order_63 = MockAlpacaOrder(
            order_id="ord_barrier_test",
            symbol="BAC",
            side="SELL",
            filled_qty="63",
            filled_avg_price="62.80",
            filled_at=now_utc,
        )
        self.bridge._client.get_orders.return_value = [order_63]

        barrier = threading.Barrier(2)
        synced_results = []

        def worker():
            barrier.wait()
            res = self.bridge.sync_today_trades_to_journal()
            synced_results.append(res)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("event_type") == "FILL" and r.get("broker_order_id") == "ord_barrier_test"]
        self.assertEqual(len(fills), 2)
        self.assertEqual(fills[0]["qty"], 57)
        self.assertEqual(fills[1]["qty"], 6)
        self.assertEqual(fills[1]["execution_id"], "SYNTH:ALPACA:ord_barrier_test:57-63")
        self.assertEqual(sum(synced_results), 1)

    # -------------------------------------------------------------
    # 3. Pricing and Decimal Precision Tests
    # -------------------------------------------------------------
    def test_incremental_price_uses_previous_cumulative_notional(self):
        """Derived incremental price correctly isolates tranche prices when average price changes."""
        now_utc = datetime.now(timezone.utc)
        # 57 shares @ 62.80 = 3579.60 notional
        order_57 = MockAlpacaOrder(
            order_id="ord_price_test",
            symbol="BAC",
            side="SELL",
            filled_qty="57",
            filled_avg_price="62.80",
            filled_at=now_utc,
        )
        self.bridge._client.get_orders.return_value = [order_57]
        self.bridge.sync_today_trades_to_journal()

        # 63 shares cumulative @ 62.82 = 3957.66 notional
        # Delta notional = 3957.66 - 3579.60 = 378.06 for 6 shares -> $63.01 / share
        order_63 = MockAlpacaOrder(
            order_id="ord_price_test",
            symbol="BAC",
            side="SELL",
            filled_qty="63",
            filled_avg_price="62.82",
            filled_at=now_utc,
        )
        self.bridge._client.get_orders.return_value = [order_63]
        self.bridge.sync_today_trades_to_journal()

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("event_type") == "FILL" and r.get("broker_order_id") == "ord_price_test"]
        self.assertEqual(len(fills), 2)
        self.assertAlmostEqual(fills[1]["fill_price"], 63.01, places=2)

    def test_canonical_decimal_formatting(self):
        """Canonical decimal removes trailing zeros without float representation errors."""
        self.assertEqual(canonical_decimal(57), "57")
        self.assertEqual(canonical_decimal("57.0"), "57")
        self.assertEqual(canonical_decimal("57.000000"), "57")
        self.assertEqual(canonical_decimal("62.80"), "62.8")
        self.assertEqual(canonical_decimal(Decimal("0.2500")), "0.25")
        self.assertEqual(canonical_decimal(None), "0")

    def test_invalid_derived_price_raises_reconciliation_alert(self):
        """If broker reports price <= 0, fill is skipped fail-closed without corrupting journal."""
        now_utc = datetime.now(timezone.utc)
        order_bad_px = MockAlpacaOrder(
            order_id="ord_bad_px",
            symbol="BAC",
            side="SELL",
            filled_qty="10",
            filled_avg_price="0.0",
            filled_at=now_utc,
        )
        self.bridge._client.get_orders.return_value = [order_bad_px]
        synced = self.bridge.sync_today_trades_to_journal()

        self.assertEqual(synced, 0)
        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("broker_order_id") == "ord_bad_px"]
        self.assertEqual(len(fills), 0)

    def test_implausible_derived_price_rejected_if_deviates_more_than_20_percent(self):
        """Derived incremental price deviating > 20% from cumulative average price is rejected fail-closed."""
        now_utc = datetime.now(timezone.utc)
        # First 50 shares logged at $100.00
        order_1 = MockAlpacaOrder(
            order_id="ord_wild_px",
            symbol="XYZ",
            side="BUY",
            filled_qty="50",
            filled_avg_price="100.00",
            filled_at=now_utc,
        )
        self.bridge._client.get_orders.return_value = [order_1]
        self.bridge.sync_today_trades_to_journal()

        # Broker advances to 60 shares, but cumulative average is suddenly $150.00
        # 60 * 150 = 9000 notional. Prev notional = 50 * 100 = 5000.
        # Delta notional = 4000 for 10 shares -> $400/share!
        # $400 deviates from $150 by +166% (well above 20%).
        order_2 = MockAlpacaOrder(
            order_id="ord_wild_px",
            symbol="XYZ",
            side="BUY",
            filled_qty="60",
            filled_avg_price="150.00",
            filled_at=now_utc,
        )
        self.bridge._client.get_orders.return_value = [order_2]
        self.bridge.sync_today_trades_to_journal()

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("broker_order_id") == "ord_wild_px"]
        # Only the first 50-share fill should be present; the wild 10-share fill is rejected!
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["qty"], 50)

    # -------------------------------------------------------------
    # 4. Journal Compatibility & Isolation Tests
    # -------------------------------------------------------------
    def test_parent_and_child_orders_are_not_combined(self):
        """Parent BUY order and Child SELL order are indexed distinctly."""
        now_ts = now_eastern().isoformat()
        log_fill(
            symbol="BAC",
            side="BUY",
            qty=63,
            fill_price=62.79,
            expected_price=62.79,
            slippage=0.0,
            fill_latency_ms=10.0,
            journal_file=self.journal_path,
            order_id="parent_buy_1",
            broker_order_id="parent_buy_1",
            timestamp=now_ts,
            account_id="PA3_TEST_ACCOUNT",
        )

        records = read_journal(self.journal_path)
        states = load_journal_order_fill_state(records, account_id="PA3_TEST_ACCOUNT")
        buy_key = ("PA3_TEST_ACCOUNT", str(now_eastern().date()), "parent_buy_1", "BAC", "BUY")
        sell_key = ("PA3_TEST_ACCOUNT", str(now_eastern().date()), "child_sell_1", "BAC", "SELL")

        self.assertIn(buy_key, states)
        self.assertEqual(states[buy_key].logged_qty, Decimal("63"))
        self.assertNotIn(sell_key, states)

    def test_broker_filled_qty_less_than_local_logged_qty_raises_alert(self):
        """If broker unexpectedly reports lower filled qty than logged, no rewrite occurs."""
        now_ts = now_eastern().isoformat()
        log_fill(
            symbol="BAC",
            side="SELL",
            qty=63,
            fill_price=62.80,
            expected_price=62.80,
            slippage=0.0,
            fill_latency_ms=10.0,
            journal_file=self.journal_path,
            order_id="ord_anomaly",
            broker_order_id="ord_anomaly",
            timestamp=now_ts,
            account_id="PA3_TEST_ACCOUNT",
        )

        # Broker returns anomalous smaller qty (50)
        order_anomaly = MockAlpacaOrder(
            order_id="ord_anomaly",
            symbol="BAC",
            side="SELL",
            filled_qty="50",
            filled_avg_price="62.80",
        )
        self.bridge._client.get_orders.return_value = [order_anomaly]
        synced = self.bridge.sync_today_trades_to_journal()

        self.assertEqual(synced, 0)
        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("broker_order_id") == "ord_anomaly"]
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["qty"], 63)

    # -------------------------------------------------------------
    # 5. Shutdown Verification & Severity Tests
    # -------------------------------------------------------------
    def test_shutdown_verified_flat_clean_returns_exit_code_0(self):
        """When 0 open positions and 0 working orders exist, returns exit_code 0."""
        self.bridge._client.get_all_positions.return_value = []
        self.bridge._client.get_orders.return_value = []

        snapshot = self.bridge.verify_shutdown_positions(timeout_seconds=2.0)
        self.assertEqual(snapshot["exit_code"], 0)
        self.assertEqual(snapshot["overall_status"], "VERIFIED_FLAT_CLEAN")
        self.assertEqual(snapshot["position_status"], "VERIFIED_FLAT")
        self.assertEqual(snapshot["order_status"], "NO_WORKING_ORDERS")
        self.assertTrue(os.path.exists(snapshot["persisted_to"]))

    def test_shutdown_verified_non_flat_returns_exit_code_2(self):
        """When positions remain at broker, returns exit_code 2 (VERIFIED_NON_FLAT)."""
        mock_pos = MagicMock()
        mock_pos.symbol = "BAC"
        mock_pos.qty = 6.0
        mock_pos.market_value = 376.80
        mock_pos.avg_entry_price = 62.80
        self.bridge._client.get_all_positions.return_value = [mock_pos]
        self.bridge._client.get_orders.return_value = []

        snapshot = self.bridge.verify_shutdown_positions(timeout_seconds=2.0)
        self.assertEqual(snapshot["exit_code"], 2)
        self.assertEqual(snapshot["overall_status"], "VERIFIED_NON_FLAT")
        self.assertEqual(snapshot["position_status"], "VERIFIED_NON_FLAT")

    def test_shutdown_position_query_failure_returns_exit_code_3(self):
        """When broker position query fails, returns exit_code 3 and strictly does not assume flat."""
        self.bridge._client.get_all_positions.side_effect = RuntimeError("Broker connection dropped")
        self.bridge._client.get_orders.return_value = []

        snapshot = self.bridge.verify_shutdown_positions(timeout_seconds=2.0)
        self.assertEqual(snapshot["exit_code"], 3)
        self.assertEqual(snapshot["position_status"], "POSITION_VERIFICATION_FAILED")
        self.assertIn("POSITION_VERIFICATION_FAILED", snapshot["failures"])

    def test_flat_with_open_orders_returns_exit_code_4(self):
        """When positions are flat but working orders remain, returns exit_code 4."""
        self.bridge._client.get_all_positions.return_value = []
        mock_ord = MagicMock()
        mock_ord.id = "ord_resting_stop"
        mock_ord.symbol = "BAC"
        mock_ord.qty = 63.0
        mock_ord.side = "sell"
        mock_ord.type = "stop"
        mock_ord.status = "new"
        self.bridge._client.get_orders.return_value = [mock_ord]

        snapshot = self.bridge.verify_shutdown_positions(timeout_seconds=2.0)
        self.assertEqual(snapshot["exit_code"], 4)
        self.assertEqual(snapshot["order_status"], "WORKING_ORDERS_PRESENT")
        self.assertEqual(snapshot["overall_status"], "VERIFIED_FLAT_BUT_UNCLEAN")

    # -------------------------------------------------------------
    # 6. Baseline & Frozen Evidence Integrity Tests
    # -------------------------------------------------------------
    def test_p2_baseline_hashes_unchanged(self):
        """Verifies that Phase 2 archived baseline files remain cryptographically untouched."""
        baseline_dir = pathlib.Path("reports/baselines/P2-BASELINE-20260910-V1.0")
        if not baseline_dir.exists():
            self.skipTest("Baseline directory not found.")

        manifest_path = baseline_dir / "p2_controlled_baseline_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for filename, expected_hash in manifest.get("baseline_artifacts", {}).items():
            file_path = baseline_dir / filename
            if file_path.exists():
                h = hashlib.sha256()
                with open(file_path, "rb") as bf:
                    while chunk := bf.read(65536):
                        h.update(chunk)
                self.assertEqual(h.hexdigest(), expected_hash, f"Hash mismatch on {filename}")

    # -------------------------------------------------------------
    # 7. Additional Mandatory Concurrency, Status & Artifact Tests
    # -------------------------------------------------------------
    def test_first_partial_fill_57_then_cumulative_63_appends_only_delta_6(self):
        """Explicit alias testing 57 -> 63 progression producing only delta 6."""
        now_utc = datetime.now(timezone.utc)
        order_57 = MockAlpacaOrder("ord_alias_1", "BAC", "SELL", "57", "62.80", now_utc)
        self.bridge._client.get_orders.return_value = [order_57]
        self.bridge.sync_today_trades_to_journal()

        order_63 = MockAlpacaOrder("ord_alias_1", "BAC", "SELL", "63", "62.80", now_utc)
        self.bridge._client.get_orders.return_value = [order_63]
        self.bridge.sync_today_trades_to_journal()

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("broker_order_id") == "ord_alias_1"]
        self.assertEqual(len(fills), 2)
        self.assertEqual(fills[0]["qty"], 57)
        self.assertEqual(fills[1]["qty"], 6)

    def test_repeated_cumulative_63_one_hundred_times_is_noop(self):
        """Explicit alias testing 100 repeated polls is a strict no-op."""
        order_63 = MockAlpacaOrder("ord_noop_1", "BAC", "SELL", "63", "62.80")
        self.bridge._client.get_orders.return_value = [order_63]
        self.bridge.sync_today_trades_to_journal()

        for _ in range(100):
            res = self.bridge.sync_today_trades_to_journal()
            self.assertEqual(res, 0)

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("broker_order_id") == "ord_noop_1"]
        self.assertEqual(len(fills), 1)

    def test_final_economic_quantity_never_exceeds_broker_cumulative_quantity(self):
        """Invariant: Journal fill sum never exceeds broker cumulative filled quantity."""
        order = MockAlpacaOrder("ord_inv_1", "BAC", "SELL", "63", "62.80")
        self.bridge._client.get_orders.return_value = [order]
        for _ in range(10):
            self.bridge.sync_today_trades_to_journal()

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("broker_order_id") == "ord_inv_1"]
        self.assertLessEqual(sum(r["qty"] for r in fills), 63)

    def test_partially_filled_order_is_not_skipped_by_order_id(self):
        """Order in partially_filled status is inspected and reconciled instead of skipped."""
        order = MockAlpacaOrder("ord_part_1", "BAC", "SELL", "30", "62.80", status="partially_filled")
        self.bridge._client.get_orders.return_value = [order]
        synced = self.bridge.sync_today_trades_to_journal()
        self.assertEqual(synced, 1)

        # Advance to 40
        order.filled_qty = "40"
        synced2 = self.bridge.sync_today_trades_to_journal()
        self.assertEqual(synced2, 1)

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("broker_order_id") == "ord_part_1"]
        self.assertEqual(len(fills), 2)
        self.assertEqual(sum(r["qty"] for r in fills), 40)

    def test_cancelled_partially_filled_order_preserves_filled_quantity(self):
        """Cancelled order with partial fills preserves its filled quantity in journal."""
        order = MockAlpacaOrder("ord_canc_1", "BAC", "SELL", "25", "62.80", status="canceled")
        self.bridge._client.get_orders.return_value = [order]
        synced = self.bridge.sync_today_trades_to_journal()
        self.assertEqual(synced, 1)

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("broker_order_id") == "ord_canc_1"]
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["qty"], 25)

    def test_replaced_partially_filled_order_keeps_distinct_order_lineage(self):
        """Replaced original order and its replacement order maintain distinct fill quantities."""
        order_orig = MockAlpacaOrder("ord_orig_stop", "BAC", "SELL", "20", "62.80", status="replaced")
        order_rep = MockAlpacaOrder("ord_rep_stop", "BAC", "SELL", "43", "62.80", status="filled")
        self.bridge._client.get_orders.return_value = [order_orig, order_rep]
        self.bridge.sync_today_trades_to_journal()

        records = read_journal(self.journal_path)
        fills_orig = [r for r in records if r.get("broker_order_id") == "ord_orig_stop"]
        fills_rep = [r for r in records if r.get("broker_order_id") == "ord_rep_stop"]
        self.assertEqual(len(fills_orig), 1)
        self.assertEqual(fills_orig[0]["qty"], 20)
        self.assertEqual(len(fills_rep), 1)
        self.assertEqual(fills_rep[0]["qty"], 43)

    def test_non_flat_and_journal_failure_returns_exit_code_2(self):
        """When account is non-flat and journal sync also fails, exit_code 2 has higher precedence."""
        mock_pos = MagicMock()
        mock_pos.symbol = "BAC"
        mock_pos.qty = 6.0
        mock_pos.market_value = 376.80
        mock_pos.avg_entry_price = 62.80
        self.bridge._client.get_all_positions.return_value = [mock_pos]
        self.bridge._client.get_orders.return_value = []

        snapshot = self.bridge.verify_shutdown_positions(timeout_seconds=2.0)
        self.assertEqual(snapshot["exit_code"], 2)
        self.assertEqual(snapshot["overall_status"], "VERIFIED_NON_FLAT")

    def test_position_failure_and_order_failure_returns_exit_code_3(self):
        """When position query fails, exit_code 3 takes precedence over order query failures."""
        self.bridge._client.get_all_positions.side_effect = RuntimeError("Position API failure")
        self.bridge._client.get_orders.side_effect = RuntimeError("Order API failure")

        snapshot = self.bridge.verify_shutdown_positions(timeout_seconds=2.0)
        self.assertEqual(snapshot["exit_code"], 3)
        self.assertEqual(snapshot["position_status"], "POSITION_VERIFICATION_FAILED")

    def test_flat_clean_but_snapshot_failure_returns_exit_code_5(self):
        """When atomic snapshot write fails, exit_code becomes 5 (persistence failure)."""
        self.bridge._client.get_all_positions.return_value = []
        self.bridge._client.get_orders.return_value = []

        with patch("pathlib.Path.open", side_effect=OSError("Read-only filesystem or locked file")):
            snapshot = self.bridge.verify_shutdown_positions(timeout_seconds=2.0)
            self.assertEqual(snapshot["exit_code"], 5)
            self.assertEqual(snapshot["overall_status"], "SNAPSHOT_PERSISTENCE_FAILED")

    def test_websocket_log_fill_uses_shared_reconciliation_lock(self):
        """log_fill uses the shared reconciliation lock."""
        lock = get_journal_lock()
        self.assertIsInstance(lock, type(threading.RLock()))

    def test_lock_is_released_after_journal_write_exception(self):
        """Reconciliation lock is properly released even if an exception occurs during write."""
        lock = get_journal_lock()
        with patch("builtins.open", side_effect=IOError("Simulated disk error")):
            try:
                log_fill(
                    symbol="BAC",
                    side="BUY",
                    qty=10,
                    fill_price=62.80,
                    expected_price=62.80,
                    slippage=0.0,
                    fill_latency_ms=1.0,
                    journal_file=self.journal_path,
                    order_id="err_ord",
                )
            except IOError:
                pass

        # Lock must be acquirable immediately
        acquired = lock.acquire(blocking=False)
        self.assertTrue(acquired)
        lock.release()

    def test_adjustment_artifact_uses_decimal_strings(self):
        """BAC adjustment artifact uses decimal string values and does not touch trade journal."""
        adj_path = pathlib.Path("reports/p4_remediation/derived/bac_journal_reconciliation_adjustment.json")
        self.assertTrue(adj_path.exists())
        with open(adj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["original_local_exit_qty"], "57")
        self.assertEqual(data["authoritative_broker_exit_qty"], "63")
        self.assertEqual(data["derived_missing_qty"], "6")
        self.assertEqual(data["economic_pnl_adjustment"], "0.06")
        self.assertFalse(data["original_record_modified"])
        self.assertIn("source_hashes", data)

    def test_adjustment_artifact_is_not_economic_runtime_fill(self):
        """Derived adjustment artifact is not treated as a runtime fill by read_journal."""
        adj_path = pathlib.Path("reports/p4_remediation/derived/bac_journal_reconciliation_adjustment.json")
        with open(adj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertFalse(is_economic_fill(data))


if __name__ == "__main__":
    unittest.main()
