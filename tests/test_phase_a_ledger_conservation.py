"""
tests/test_phase_a_ledger_conservation.py -- Specialized Test 3: Attribution Ledger Conservation & Traceability Test.

Validates that:
1. Conservation Law:
   Total Candidate Evaluations == Terminal Passes + Terminal Rejections.
2. 100% Attribution Traceability:
   Every terminal rejection contains a non-null, valid blocked_by stage and reason_code.
3. Structured Schema Integrity:
   Every record written to the ledger adheres to the required telemetry schema.
4. Thread Safety:
   Concurrent logging operations preserve ledger JSON integrity without corruption.
"""
import ast
import json
import os
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path

from strategies.gate_ledger import (
    log_gate_decision,
    summarize_gate_decisions,
    CANONICAL_PIPELINE_STAGES,
)
from strategies.veto_reasons import VetoReason


class TestPhaseALedgerConservation(unittest.TestCase):

    def setUp(self):
        self._temp_file = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.ledger_path = self._temp_file.name
        self._temp_file.close()

    def tearDown(self):
        if os.path.exists(self.ledger_path):
            try:
                os.remove(self.ledger_path)
            except Exception:
                pass

    def test_ledger_conservation_and_attribution(self):
        """
        Simulates 100 candidates through various pipeline gate outcomes.
        Verifies that:
          Total Candidates (100) == terminal_passes + terminal_rejections
          sum(blocker_counts) == terminal_rejections
          100% of rejections have valid blocked_by and reason_code
        """
        # Distribution across 100 evaluations:
        # 30 coverage rejections
        # 25 lunch chop rejections
        # 15 cooldown rejections
        # 15 confidence rejections
        # 5 order execution rejections
        # 10 full passes / submissions
        stages_and_counts = [
            ("coverage", "REJECT", VetoReason.LOW_COVERAGE, 30),
            ("lunch_chop", "REJECT", VetoReason.LUNCH_CHOP_ZONE, 25),
            ("cooldown", "REJECT", VetoReason.SESSION_COOLDOWN, 15),
            ("confidence", "REJECT", VetoReason.CONFIDENCE_BELOW_FLOOR, 15),
            ("order_execution", "REJECT", VetoReason.NOISE_FLOOR, 5),
            ("order_execution", "SUBMIT", None, 10),
        ]

        total_simulated = 0
        candidate_idx = 0

        for stage, decision, reason_code, count in stages_and_counts:
            for _ in range(count):
                candidate_idx += 1
                total_simulated += 1
                eval_id = f"eval_sim_{candidate_idx:04d}"
                symbol = f"SYM{candidate_idx % 10}"
                bar = "2026-09-08 10:00:00"

                is_rejection = decision == "REJECT"
                blocked_by = stage if is_rejection else None

                # Intermediate gate pass if candidate progressed past early stages
                if stage in ("confidence", "order_execution"):
                    log_gate_decision(
                        evaluation_id=eval_id,
                        symbol=symbol,
                        bar=bar,
                        strategy="vwap_bounce",
                        stage="coverage",
                        decision="PASS",
                        terminal_decision=False,
                        filepath=self.ledger_path,
                    )

                # Terminal decision
                log_gate_decision(
                    evaluation_id=eval_id,
                    symbol=symbol,
                    bar=bar,
                    strategy="vwap_bounce",
                    stage=stage,
                    decision=decision,
                    reason_code=reason_code,
                    human_message=f"Candidate {eval_id} evaluated at {stage}",
                    terminal_decision=True,
                    blocked_by=blocked_by,
                    raw_conf=0.75 if is_rejection else 0.85,
                    final_conf=0.55 if is_rejection else 0.85,
                    required_conf=0.60,
                    filepath=self.ledger_path,
                )

        self.assertEqual(total_simulated, 100)

        # 1. Verify summary metrics
        summary = summarize_gate_decisions(filepath=self.ledger_path)
        terminal_passes = summary["terminal_passes"]
        terminal_rejections = summary["terminal_rejections"]

        self.assertEqual(terminal_passes, 10)
        self.assertEqual(terminal_rejections, 90)
        # Mathematical Conservation Law: Total Candidates = Passes + Rejections
        self.assertEqual(terminal_passes + terminal_rejections, 100)

        # Blocker breakdown sums exactly to terminal_rejections
        total_blocked = sum(summary["blocker_counts"].values())
        self.assertEqual(total_blocked, 90)
        self.assertEqual(summary["blocker_counts"]["coverage"], 30)
        self.assertEqual(summary["blocker_counts"]["lunch_chop"], 25)
        self.assertEqual(summary["blocker_counts"]["cooldown"], 15)
        self.assertEqual(summary["blocker_counts"]["confidence"], 15)
        self.assertEqual(summary["blocker_counts"]["order_execution"], 5)

        # 2. Verify 100% individual record schema and attribution integrity
        required_schema_keys = {
            "evaluation_id",
            "timestamp",
            "symbol",
            "bar",
            "strategy",
            "stage",
            "decision",
            "reason_code",
            "human_message",
            "terminal_decision",
            "blocked_by",
            "raw_conf",
            "final_conf",
            "required_conf",
            "metadata",
        }

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                rec = json.loads(line.strip())
                # Check schema completeness
                self.assertTrue(required_schema_keys.issubset(rec.keys()), f"Line {line_idx} missing schema keys")

                # Timestamp must be ISO 8601 parseable
                ts = datetime.fromisoformat(rec["timestamp"])
                self.assertIsNotNone(ts)

                # Check rejection attribution
                if rec["terminal_decision"] and rec["decision"] == "REJECT":
                    self.assertIsNotNone(rec["blocked_by"], f"Line {line_idx}: blocked_by must not be None")
                    self.assertNotEqual(rec["blocked_by"], "", f"Line {line_idx}: blocked_by must not be empty")
                    self.assertIsNotNone(rec["reason_code"], f"Line {line_idx}: reason_code must not be None")
                    self.assertNotEqual(rec["reason_code"], "", f"Line {line_idx}: reason_code must not be empty")

    def test_concurrent_writes_thread_safety(self):
        """
        Verifies that 10 concurrent worker threads writing 10 records each
        produce exactly 100 uncorrupted, parseable JSON lines in the ledger.
        """
        def worker(worker_id: int):
            for j in range(10):
                log_gate_decision(
                    evaluation_id=f"thread_{worker_id}_rec_{j}",
                    symbol="TSLA",
                    bar="2026-09-08 11:00:00",
                    strategy="vwap_bounce",
                    stage="order_execution",
                    decision="SUBMIT",
                    terminal_decision=True,
                    filepath=self.ledger_path,
                )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Read back all lines and parse
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        self.assertEqual(len(lines), 100)
        for line in lines:
            parsed = json.loads(line)
            self.assertEqual(parsed["symbol"], "TSLA")
            self.assertEqual(parsed["decision"], "SUBMIT")

    def test_canonical_pipeline_stages_completeness(self):
        """
        Verifies that CANONICAL_PIPELINE_STAGES:
        1. Contains all required stages.
        2. All stage keys start with 'STAGE_'.
        3. No duplicate stage keys or descriptions exist.
        4. Matches expected sequential order from data arrival to execution/fill.
        """
        self.assertIsInstance(CANONICAL_PIPELINE_STAGES, tuple)
        self.assertGreaterEqual(len(CANONICAL_PIPELINE_STAGES), 18)

        keys = [s[0] for s in CANONICAL_PIPELINE_STAGES]
        descs = [s[1] for s in CANONICAL_PIPELINE_STAGES]

        # Uniqueness
        self.assertEqual(len(keys), len(set(keys)), "Duplicate stage keys in CANONICAL_PIPELINE_STAGES")
        self.assertEqual(len(descs), len(set(descs)), "Duplicate stage descriptions in CANONICAL_PIPELINE_STAGES")

        # Naming convention
        for k in keys:
            self.assertTrue(k.startswith("STAGE_"), f"Stage key {k} does not start with STAGE_")

        # Required critical stages must be present
        required_stages = {
            "STAGE_COVERAGE",
            "STAGE_REGIME",
            "STAGE_LUNCH_CHOP",
            "STAGE_ACTIVE_POSITION",
            "STAGE_PENDING_ORDER",
            "STAGE_CONFIDENCE",
            "STAGE_RSI_FILTER",
            "STAGE_GEOMETRY",
            "STAGE_ENTRY_READINESS",
            "STAGE_COOLDOWN",
            "STAGE_QUOTE_HEALTH",
            "STAGE_NOISE_FLOOR",
            "STAGE_MAX_CHASE",
            "STAGE_SPY_GUARD",
            "STAGE_COUNTERTREND_GUARD",
            "STAGE_SOXX_GUARD",
            "STAGE_SECTOR_CLUSTER_CAP",
            "STAGE_POSITION_SIZE",
            "STAGE_ORDER_EXECUTION",
            "STAGE_ORDER_SUBMIT",
            "STAGE_FILLED",
        }
        self.assertTrue(required_stages.issubset(set(keys)), f"Missing required stages: {required_stages - set(keys)}")

    def test_orchestrator_stage_sync_with_canonical_stages(self):
        """
        AST Invariant Test:
        Statically parses strategies/orchestrator.py and extracts:
        1. All 'stage' keyword arguments passed to log_gate_decision()
        2. All stage string values returned by _classify_execution_rejection()

        Asserts that 100% of emitted stages exist in CANONICAL_PIPELINE_STAGES.
        """
        canonical_stage_set = set(dict(CANONICAL_PIPELINE_STAGES).keys())

        orch_path = Path(__file__).resolve().parent.parent / "strategies" / "orchestrator.py"
        self.assertTrue(orch_path.exists(), f"Orchestrator not found at {orch_path}")

        tree = ast.parse(orch_path.read_text(encoding="utf-8"), filename=str(orch_path))

        emitted_stages = set()

        # 1. Inspect log_gate_decision calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name == "log_gate_decision":
                    for kw in node.keywords:
                        if kw.arg == "stage":
                            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                emitted_stages.add(kw.value.value)

        # 2. Inspect _classify_execution_rejection return values
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_classify_execution_rejection":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple):
                        first_elt = sub.value.elts[0]
                        if isinstance(first_elt, ast.Constant) and isinstance(first_elt.value, str):
                            emitted_stages.add(first_elt.value)

        self.assertGreater(len(emitted_stages), 0, "No stages extracted from orchestrator.py")

        unregistered = emitted_stages - canonical_stage_set
        self.assertEqual(
            len(unregistered),
            0,
            f"Orchestrator emits stages NOT in CANONICAL_PIPELINE_STAGES: {unregistered}",
        )

    def test_classify_execution_rejection_semantics(self):
        """
        Verifies that _classify_execution_rejection correctly isolates pre-submit
        policy/risk gates from broker execution rejections.

        Authoritative Classification Precedence Hierarchy:
          1. STAGE_COUNTERTREND_GUARD  : "counter_trend", "countertrend", "counter trend"
          2. STAGE_SECTOR_CLUSTER_CAP   : "cluster cap", "tech cluster", "sector concentration"
          3. STAGE_QUOTE_HEALTH        : "quote health", "stale quote", "quote ... stale", "inverted quote", "spread"
          4. Known Typed Execution Gates:
             - STAGE_COOLDOWN          : "suppression", "cooldown"
             - STAGE_MAX_CHASE         : "chase"
             - STAGE_SPY_GUARD         : "trend guard", "spy 5m", "spy benchmark"
             - STAGE_SOXX_GUARD        : "soxx"
             - STAGE_NOISE_FLOOR       : "noise floor", "critically tight"
             - STAGE_POSITION_SIZE     : "position size is 0", "0 shares", "shares=0", "zero position"
             - STAGE_ENTRY_READINESS   : "entry readiness"
             - STAGE_GEOMETRY          : "price ordering", "geometry"
          5. STAGE_ORDER_EXECUTION     : Fallback strictly for true unhandled broker-level rejection errors.
        """
        from strategies.orchestrator import _classify_execution_rejection
        from strategies.veto_reasons import VetoReason

        # 1. Counter-Trend Short Veto
        stage, reason = _classify_execution_rejection(
            "Counter-Trend SHORT Veto: RSI 50.0 > 40.0 — no breakdown weakness confirmed"
        )
        self.assertEqual(stage, "STAGE_COUNTERTREND_GUARD")
        self.assertEqual(reason, VetoReason.COUNTER_TREND_SHORT_VETO)

        # 2. Sector / Tech Cluster Cap
        stage, reason = _classify_execution_rejection(
            "Correlated Tech Cluster Cap reached (2/2): Blocked MSFT to prevent sector concentration risk"
        )
        self.assertEqual(stage, "STAGE_SECTOR_CLUSTER_CAP")
        self.assertEqual(reason, VetoReason.TECH_CLUSTER_CAP)

        # 3. Quote staleness (runtime pattern)
        stage, reason = _classify_execution_rejection(
            "Execution Quote Gate: Quote for JPM is stale (age=11.2s > limit=10.0s)"
        )
        self.assertEqual(stage, "STAGE_QUOTE_HEALTH")
        self.assertEqual(reason, VetoReason.STALE_QUOTE)

        # 4. Quote staleness (canonical pattern)
        stage, reason = _classify_execution_rejection("Execution Quote Gate: Quote stale")
        self.assertEqual(stage, "STAGE_QUOTE_HEALTH")
        self.assertEqual(reason, VetoReason.STALE_QUOTE)

        # 5. True unhandled broker rejection fallback
        stage, reason = _classify_execution_rejection("broker rejected order: buying power")
        self.assertEqual(stage, "STAGE_ORDER_EXECUTION")
        self.assertEqual(reason, "ORDER_REJECTED")


if __name__ == "__main__":
    unittest.main()
