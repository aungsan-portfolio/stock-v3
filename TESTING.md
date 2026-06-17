# Testing & Verification

> ⚠️ **Paper-trading only.** This engine targets IBKR **paper** accounts. It is a
> research/education tool. Past performance does not guarantee future results.
> Do not point it at a live, real-money account.

## Dependencies

The test suite and verification script use only Python's standard-library
`unittest` — no pytest, no lint/type tooling. Installing the runtime
dependencies is enough:

```bash
pip install -r requirements-dev.txt   # pulls in -r requirements.txt
```

## Running the unit tests

```bash
python -m unittest test_logic -v             # core refactor logic
python -m unittest test_live_readiness -v    # Phase-0 live-readiness scaffolding
python -m unittest test_model_gate -v        # Phase-1 model gate + signal parity
python -m unittest test_order_exec -v        # Phase-2 order-execution logic (H4-H9)
python -m unittest test_ibkr_fill_flow -v    # Phase-2 fill-driven order path (fake IBKR)
python -m unittest test_risk_engine -v       # Phase-3 risk engine + equity snapshot (H1, H3)
python -m unittest test_phase3_protection -v # Phase-3 GTC/OCA + kill-switch + startup invariant
python -m unittest test_reconciliation -v    # Phase-5A startup reconciliation (broker = source of truth)
python -m unittest discover -p "test_*.py"   # everything
python test_logic.py                         # equivalent direct run
```

The tests are **deterministic and offline**: they feed synthetic OHLCV data and
stub the per-symbol model predictions, so no network access or trained model
files are required. Backtest tests redirect `REPORTS_DIR` to a temp directory,
so they never overwrite real reports.

### What the tests cover

| Area | Coverage |
|---|---|
| `apply_position_rule_with_hold` | long-only open/hold/close, short-enabled open/cover, no single-bar flip |
| `min_hold` guard | opposite signals ignored until `MIN_HOLD_BARS`; exact hold-period |
| Ensemble safety | both ML models missing → forced `HOLD`; one model present → real ensemble |
| Config | `MIN_HOLD_BARS == ML_HORIZON` |
| Backtest I/O | empty-data → empty CSV + metrics JSON; populated → CSV rows + schema |
| Imports | `main`, `backtest`, `predictor`, `ibkr_bridge`, engines import cleanly |
| Live-readiness (Phase 0) | `unprotected_longs` / duplicate-ref invariants; `order_audit` JSONL never raises; Phase-3+ bridge capability flags `False` (Phase-2 flags now `True`); `live-readiness` scorecard reports NOT READY (`test_live_readiness.py`) |
| Order execution (Phase 2) | fill-vs-accepted classification (`PreSubmitted` is not a fill); partial-fill exit-child reconciliation — **all** SELL children (protective stop, trailing stop, AND take-profit LMT) cancelled/resized to the actual filled qty so nothing can over-sell into a short; protective-child verify-or-flatten; robust close marketable-limit→market escalation; deterministic `orderRef` duplicate guard; emergency flatten cancels resting children first; capability flags (`test_order_exec.py`, `test_ibkr_fill_flow.py`) |
| Server-side protection + risk (Phase 3) | GTC + OCA exit set placed **after** the confirmed fill, sized to the filled qty; independent hard −3% stop priced off the **actual** `avgFillPrice`; every exit `tif=GTC` and shares one non-empty OCA group; no exit qty exceeds the fill; placement/verify failure → flatten + abort; daily-loss kill-switch (start-of-day equity snapshot) blocks **new opens** but allows **closes**; account drawdown / per-symbol exposure groundwork; startup invariant repairs unprotected longs or halts new entries; capability flags (`test_order_exec.py`, `test_risk_engine.py`, `test_phase3_protection.py`) |
| Startup reconciliation (Phase 5A) | `reconciliation.build_snapshot` broker-truth snapshot from plain broker state — protected vs. unprotected longs (GTC-aware: a DAY stop or take-profit LMT is not protection), duplicate `orderRef` detection, orphan exit orders (resting SELL with no covering long); `IBKRBridge.reconcile_startup_state()` driven through the real bridge with fake IBKR proves **broker = source of truth**, repairs an unprotected long via the existing Phase-3 path or halts new entries, audits the snapshot, and **never opens a new position** (`test_reconciliation.py`) |
| Model gate (Phase 1) | `evaluate_gate` decision table (missing / stale / below-floor / ok / disabled); per-symbol metrics persist + merge RF and LSTM and never raise; a pre-gate BUY is forced to `HOLD` when blocked; backtest/live signal-construction parity; P1 scorecard gates PASS while overall stays NOT READY; `signal-parity` command (`test_model_gate.py`) |

> **Model gate & signal parity (Phase 1).** `test_model_gate.py` is offline and
> patches `config.MODEL_METRICS_FILE` to a temp file, so it never reads or writes
> the real `models/model_metrics.json`. The gate is **fail-closed**: a symbol
> with missing, stale, or below-floor metrics is forced to `HOLD`. Two tests
> deliberately exercise the corrupt-file / bad-input recovery paths, which log a
> warning (with traceback) and return safe defaults — those tracebacks in the
> test output are expected and the tests still pass. Capability flags
> (`SUPPORTS_MODEL_PERFORMANCE_GATE`, `SUPPORTS_MODEL_STALENESS_GATE`) are `True`
> only because the logic exists **and** is covered here.

> **Live-readiness scaffolding (Phase 0).** `test_live_readiness.py` is offline.
> The optional broker integration tests (paper TWS required) are skipped unless
> you set `RUN_IBKR_INTEGRATION=1` with TWS running on port 7497. See
> `reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md` for the phased plan; live
> trading stays disabled until every automated `live-readiness` gate passes.

> **Order execution robustness (Phase 2, H4-H9).** `test_order_exec.py` is pure
> and offline (the ib-free decision layer in `order_exec.py`); `test_ibkr_fill_flow.py`
> drives the real `ibkr_bridge.py` order path with **fake ib_insync objects**
> (no live IBKR, no network). Together they prove the Phase-2 go/no-go gates:
> an accepted-but-unfilled order is **not** recorded as a trade; a partial fill
> reconciles **every** exit-side child — protective stop, trailing stop, AND
> take-profit LMT — down to the actual filled qty (or flattens) so a resting
> SELL can never sell more than is held; an unprotectable fill is
> emergency-flattened and marked aborted; closes escalate marketable-limit →
> market until flat; and a duplicate open after a restart is refused via a
> deterministic `orderRef`. The "Unsafe exit protection … emergency flatten" and
> "Emergency close …" lines in the test output are the **expected** logs from the
> flatten-path tests. Capability flags `SUPPORTS_FILL_VERIFICATION`,
> `SUPPORTS_PARTIAL_FILL_HANDLING`, and `SUPPORTS_PROTECTIVE_CHILD_VERIFY` are
> `True` only because the logic exists **and** is covered here; the Phase-3+ flags
> remain `False`, so the `live-readiness` scorecard still reports NOT READY.

> **Server-side protection + risk engine (Phase 3, C2/H19/H1/H3).** `test_risk_engine.py`
> is pure (drawdown / exposure math + the file-backed start-of-day equity
> snapshot, patched to a temp file); `test_phase3_protection.py` drives the real
> bridge with the fake ib_insync objects from `test_ibkr_fill_flow.py`. They prove:
> protection is placed **after** the confirmed fill, every exit is `tif=GTC` and
> shares one OCA group, an independent hard −3% stop is sized to the filled qty
> and priced off the **actual** `avgFillPrice` (not the signal price), no exit can
> over-sell, a placement/verify failure flattens + aborts, the daily-loss
> kill-switch blocks new opens while still allowing closes, and the startup
> invariant repairs an unprotected long or halts new entries (avgCost missing →
> halt, never guess). The "Unsafe GTC protection … emergency flatten",
> "New entry blocked … (daily_loss/halt_flag)", and "Startup: …" lines in the
> test output are **expected**. Flags `SUPPORTS_SERVER_SIDE_GTC_STOP` and
> `SUPPORTS_DAILY_LOSS_KILLSWITCH` flip `True`; Phase-4–6 flags stay `False`, so
> `live-readiness` still reports NOT READY.

> **Startup reconciliation (Phase 5A, H18).** `reconciliation.py` is the **pure**
> half (no ib_insync / network): it builds a broker-truth snapshot from plain
> position / working-order dicts and never mutates anything. `test_reconciliation.py`
> proves both halves: the snapshot classifies a long as protected only when a
> resting **GTC** stop covers its qty (a DAY stop or take-profit LMT is *not*
> downside protection), flags duplicate `orderRef`s and orphan exit orders, and
> the bridge method `IBKRBridge.reconcile_startup_state()` (alias
> `startup_reconcile`) treats the **broker as the source of truth** — never a
> local file — audits the snapshot (`order_audit` stage `reconcile`), and repairs
> an unprotected long **only** through the existing Phase-3 `ensure_protective_stops`
> path (or halts new entries; avgCost missing → halt, never guess). The core
> safety test asserts reconciliation **never places a BUY / opens a position**.
> The "Startup reconciliation …" and "Startup: …" lines in the test output are
> **expected**. `SUPPORTS_STARTUP_RECONCILIATION` flips `True` only because the
> logic exists **and** is covered here; the Phase-6 flag
> (`SUPPORTS_ACCOUNT_TYPE_ASSERTION`) stays `False`, so `live-readiness` still
> reports NOT READY. Phase 5B (scheduler / reconnect watchdog / graceful
> shutdown / alerting) is **not** included.

## Continuous integration

`.github/workflows/ci.yml` runs on push / pull request against a Python
**3.11** and **3.12** matrix:

1. `python -m py_compile` over every module.
2. `python -m unittest test_logic -v`.

No build artifacts are produced.

## Production verification (multi-symbol)

`verify_watchlist.py` exercises the live prediction + backtest paths over the
whole `WATCHLIST` **without placing any orders**, then writes a summary JSON to
`reports/verification_summary.json`.

```bash
python verify_watchlist.py                  # predict + full backtest
python verify_watchlist.py --skip-backtest  # faster, predict-only
```

It asserts the safety invariants below and exits `0` on success, `1` on any
violation. (Unlike the unit tests, this script fetches **live** data via
`yfinance`, so it needs network access.)

## Current production-safety behavior

These behaviors are guaranteed by the current logic and verified by the tests
and the verification script:

- **Model missing → forced `HOLD`.**
  If a symbol has no trained ML model (neither RF nor LSTM), the engine forces
  `action = HOLD`. A missing per-symbol model raises `ModelNotAvailableError`
  rather than returning a neutral `0.5`, so it can never silently count as an
  available model. Technical-only BUY/SELL trading stays disabled
  (`MIN_ML_MODELS_FOR_SIGNAL = 1`).

- **Untrusted model → forced `HOLD` (Phase 1 signal-safety gate).**
  Before any BUY/SELL is issued, `predictor.predict_all` consults
  `model_metrics.evaluate_gate(symbol)`. A symbol whose persisted metrics are
  **missing**, **stale** (older than `MODEL_MAX_AGE_DAYS`), or **below the floor**
  (`MODEL_MIN_RF_ACCURACY`, optional `MODEL_MIN_RF_F1`) is forced to `HOLD` with
  the reason appended to the signal. This is **fail-closed**: no trusted metric
  means no trade. Metrics are written per symbol at train time and the gate can
  be master-disabled with `MODEL_GATE_ENABLED` (tests only — ships `True`).

- **`MIN_HOLD_BARS = ML_HORIZON`.**
  A position is held for at least `MIN_HOLD_BARS` bars before an opposite signal
  may close it. This is set equal to `ML_HORIZON` so the held period matches the
  forward-direction horizon the models are trained on.

- **Long-only by default (`ALLOW_SHORT = False`).**
  `SELL` closes an existing long; it does not open a short unless `ALLOW_SHORT`
  is explicitly enabled. BUY/SELL never flip a position in a single bar — they
  close to flat first.
