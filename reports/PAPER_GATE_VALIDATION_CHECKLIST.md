# Paper Gate Empirical Validation Checklist (Phase 3 / 4 / 5)

**Status: PAPER ONLY. This document changes NO code and NO config.**
It is a run-book for validating the Phase 0–5B safety gates against a **paper**
TWS / IB Gateway on **port 7497**. It never enables live trading.

- Do **not** start Phase 6. Do **not** flip any flag to live.
- Required safety state (already shipped in `config.py`, re-verified — see §0):
  `REQUIRE_PAPER_PORT=True`, `IBKR_PORT=7497`, `PAPER_IBKR_PORT=7497`,
  `ALLOW_HISTORICAL_PRICE_FOR_ORDERS=False`, `LIVE_ACCOUNT_ID=None`,
  `MARKET_HOURS_GATE_ENABLED=True`, `COACH_LIVE_TRADING_ENABLED=False`.
- All orders below are **paper** orders on a paper account. No real orders.

> Run every command from the project root in **PowerShell**. Set this once per
> shell so the snippets are copy-paste ready:
>
> ```powershell
> $Root = "C:\Users\Aung San\Downloads\stock_engine_pro_v3_production_ready\stock_engine_pro_v3_production_ready"
> Set-Location $Root
> ```

---

## Where to look (logs / audit / state)

| File | What it holds | Quick read (PowerShell) |
|---|---|---|
| `logs\stock_engine.log` | Human-readable rotating log (all modules) | `Get-Content .\logs\stock_engine.log -Tail 80` |
| `logs\order_audit.jsonl` | **Machine-readable order/event audit** (one JSON per line) | `Get-Content .\logs\order_audit.jsonl -Tail 40` |
| `logs\daily_risk_state.json` | Per-day `trades` count + `start_equity` baseline | `Get-Content .\logs\daily_risk_state.json` |

Audit `stage` values you will grep for below: `equity_snapshot`, `reconcile`,
`protect`, `stop_confirmed`, `halt`, `shutdown`, `watchdog`, `schedule`,
`quote`, `filled`, `flatten`, `alert`.

Generic audit filter (works for any stage):
```powershell
Select-String -Path .\logs\order_audit.jsonl -Pattern '"stage": "reconcile"'
```

---

## 0. Preconditions (do these first)

### 0.1 Re-verify the safety state is intact (read-only — places nothing)
```powershell
python -X utf8 -c "import config; print('REQUIRE_PAPER_PORT', config.REQUIRE_PAPER_PORT); print('IBKR_PORT', config.IBKR_PORT); print('PAPER_IBKR_PORT', config.PAPER_IBKR_PORT); print('ALLOW_HISTORICAL_PRICE_FOR_ORDERS', config.ALLOW_HISTORICAL_PRICE_FOR_ORDERS); print('LIVE_ACCOUNT_ID', getattr(config,'LIVE_ACCOUNT_ID', None)); print('MARKET_HOURS_GATE_ENABLED', config.MARKET_HOURS_GATE_ENABLED); print('COACH_LIVE_TRADING_ENABLED', config.COACH_LIVE_TRADING_ENABLED); print('PROTECTIVE_TIF', config.PROTECTIVE_TIF); print('MAX_DAILY_LOSS_USD', config.MAX_DAILY_LOSS_USD); print('ALERTS_ENABLED', config.ALERTS_ENABLED); print('ALERTS_LOG_ONLY', config.ALERTS_LOG_ONLY); print('IBKR_MARKET_DATA_TYPE', config.IBKR_MARKET_DATA_TYPE)"
```
**PASS** when output is exactly:
```
REQUIRE_PAPER_PORT True
IBKR_PORT 7497
PAPER_IBKR_PORT 7497
ALLOW_HISTORICAL_PRICE_FOR_ORDERS False
LIVE_ACCOUNT_ID None
MARKET_HOURS_GATE_ENABLED True
COACH_LIVE_TRADING_ENABLED False
PROTECTIVE_TIF GTC
MAX_DAILY_LOSS_USD 150.0
ALERTS_ENABLED False
ALERTS_LOG_ONLY True
IBKR_MARKET_DATA_TYPE 3
```
**FAIL** on any deviation → stop and restore `config.py` before continuing.

### 0.2 TWS / IB Gateway paper setup (manual, in the GUI)
- Log into the **paper** account (account id starts with `DU…`).
- API → Settings → **Enable ActiveX and Socket Clients** = ON.
- **Socket port = 7497** (paper). Confirm it is *not* 7496 (live).
- "Read-Only API" = **OFF** (the bot must be able to place paper orders).
- Allow connections from `127.0.0.1`.
- Market data: the bot requests **delayed** data (`IBKR_MARKET_DATA_TYPE=3`),
  so a real-time subscription is **not** required for paper.

### 0.3 Offline gate backstop (no IBKR, no orders) — optional but recommended
The pure-logic half of every gate below is already unit-tested. Re-run the
relevant suites to confirm the build is intact before touching the broker:
```powershell
python -X utf8 -m unittest test_phase3_protection test_data_integrity test_risk_engine test_reconciliation test_reconnect_watchdog test_shutdown_guard test_alerts test_scheduler_runner -v
```
**PASS**: `OK`. These are the deterministic backstop; the steps below add the
*empirical broker-truth* layer on top.

### 0.4 Read-only go-live scorecard (no orders; expected to say NOT READY)
```powershell
python -X utf8 main.py live-readiness
```
**PASS**: prints the gate table and ends with **`[NOT READY]`** (because
`SUPPORTS_ACCOUNT_TYPE_ASSERTION=False` and `LIVE_ACCOUNT_ID=None` — the Phase-6
gates are intentionally still failing). Ends with `No IBKR orders were placed.`
This confirms the live conversion remains correctly blocked.

---

## Gate group A — one flow validates Gates 1, 2, 3, 7

The cleanest empirical test of server-side protection, broker-truth
reconciliation, unprotected-long repair, and graceful shutdown is a **single
manually-created unprotected long**. You do not need the model to emit a BUY.

> Timing tip: do the manual buy while the **market is open** so it fills. The
> protective-repair path is *not* market-hours gated, so the rest works anytime.

### A.0 Create the test condition (manual, in TWS)
1. In TWS paper, place a **market BUY of 1 share SPY**. Do **not** attach any
   stop / bracket. Let it fill.
2. Confirm in TWS: Portfolio shows **SPY +1**, and there are **no working
   orders** for SPY (it is an *unprotected* long).

### A.1 Detect the unprotected long (read-only — places nothing)
```powershell
python -X utf8 main.py live-readiness --connect
```
- **Observe (console):** a `BRK` row →
  `Every open long has a working protective stop … [FAIL] … unprotected: SPY`.
- **PASS (Gate 2 detection / Gate 3 detection):** the connected scorecard
  reports `unprotected: SPY` (it read the broker, not a local file). No order
  was placed (`No IBKR orders were placed.`).

### A.2 Reconcile + repair + graceful shutdown (places **paper** protective orders only)
```powershell
python -X utf8 main.py paper
```
This runs, in order: start-of-day equity snapshot → **startup reconciliation
(broker = source of truth)** → repair of the unprotected long → predict/execute
(may be all HOLD, that is fine) → **graceful shutdown**.

- **Observe (console):**
  - `Startup reconciliation (broker = source of truth): 1 long(s), … clean=False`
  - `Repaired protective GTC stop(s) for: ['SPY']`
- **Observe (TWS) — Gate 1 (server-side GTC/OCA stop):** SPY now has **two
  resting SELL orders**, both **TIF = GTC**, sharing one OCA group
  `OCA:<YYYY-MM-DD>:SPY`, each **qty = 1**:
  - `SELL STP` (hard stop) ≈ `avgCost × (1 − 0.03)`
  - `SELL TRAIL` (trailing 0.4%) trail-stop ≈ `avgCost × (1 − 0.004)`
- **Observe (audit):**
  ```powershell
  Select-String -Path .\logs\order_audit.jsonl -Pattern '"stage": "reconcile"','"stage": "protect"','"stage": "shutdown"' | Select-Object -Last 12
  ```
  - a `reconcile` line with `"source": "broker"`, `"unprotected_longs": ["SPY"]`
  - a `protect` line with `"phase": "startup_repair"`, `"symbol": "SPY"`,
    `"tif": "GTC"`
  - `shutdown` lines `"phase": "plan"` then `"phase": "complete"` with
    `"opens_new_entries": false`, `"n_orders_to_cancel": 0`
- **Observe (log) — Gate 7 (graceful shutdown):**
  ```powershell
  Select-String -Path .\logs\stock_engine.log -Pattern 'Graceful shutdown' | Select-Object -Last 2
  ```
  → `graceful shutdown [SAFE]: 1 long(s), … protective stop(s) preserved, 0 cancelled`

**PASS criteria for A.2:**
- **Gate 3 (repair):** console shows `Repaired protective GTC stop(s) for: ['SPY']`
  and TWS shows the new GTC stops.
- **Gate 1 (server-side stop):** the two GTC SELL orders rest in TWS with the
  shared OCA group.
- **Gate 7 (graceful shutdown preserves protection):** the shutdown line shows
  **`0 cancelled`** and `opens_new_entries=false`; the GTC stops are still there.

### A.3 The GTC stop survives the bot process exit (Gate 1, the key property)
The `paper` command has now **exited** (the Python process is gone).
- **Observe (TWS):** the two SPY **GTC** SELL orders are **still resting**.
- **PASS (Gate 1):** protection persists at the broker with the bot **not
  running** — this is the whole point of GTC server-side stops.

### A.4 Re-reconcile now reads the position as protected (Gate 2, broker truth)
```powershell
python -X utf8 main.py live-readiness --connect
```
- **PASS (Gate 2):** the `BRK` row now reads **`[PASS] … unprotected: none`** —
  reconciliation reflects the *current broker state* (the stops you can see in
  TWS), proving broker-truth, not a cached local view.

### A.5 Cleanup
```powershell
python -X utf8 main.py panic-flatten           # DRY-RUN: shows what it would cancel/flatten
python -X utf8 main.py panic-flatten --confirm  # cancels SPY GTC stops + flattens the 1 share
```
- **PASS:** TWS shows SPY flat and no working SPY orders. (`flatten` audit lines
  recorded.) Account is clean for the next gate.

---

## Gate 4 — Market-hours gate blocks NEW entries (closes/repair never gated)

The bridge gate (`_market_hours_ok` → `data_integrity.is_regular_hours`) and the
supervised scheduler share the **same** US holiday/early-close calendar.

### 4.1 Scheduler block (deterministic; run this **outside** 09:30–16:00 ET / weekend / holiday)
```powershell
python -X utf8 main.py run-scheduled
```
- **Observe (console):** `[SKIP] Scheduled run blocked: <weekend|after_close|before_open|holiday>`
  and `No orders placed. No IBKR connection made.`
- **Observe (audit):**
  ```powershell
  Select-String -Path .\logs\order_audit.jsonl -Pattern '"stage": "schedule"' | Select-Object -Last 3
  ```
  → `"decision": "blocked"`, `"reason": "<weekend|after_close|…>"`.
- **PASS:** blocked with the correct reason; **no IBKR connection, no order**.

### 4.2 Bridge entry-gate block (when a BUY is attempted outside RTH)
Run the full bot **outside RTH**:
```powershell
python -X utf8 main.py paper
```
- If today produced an actionable BUY, the entry is refused:
  - **Log:** `New entry blocked for <SYM> BUY (market_closed)`
  - **Audit:** a `halt` line `"phase": "entry_gate"`, `"reason": "market_closed"`.
- **PASS:** every NEW entry outside RTH is blocked with `reason=market_closed`,
  while startup reconciliation / protective repair / closes still run.
- **Note:** if the model emits only HOLD that day, the entry-gate code path is
  not reached; 4.1 + the `test_data_integrity` suite are the deterministic proof
  that the calendar logic is correct. (Optional on-demand method: §Appendix P.)

> Inverse check (optional): the same `run-scheduled` **inside** RTH on a trading
> day prints `[RUN] Scheduled run allowed` and dispatches the paper command.

---

## Gate 5 — Daily-loss kill-switch blocks NEW entries

`risk_state.daily_loss_blocked(equity)` trips when
`start_of_day_equity − current_equity ≥ MAX_DAILY_LOSS_USD` (150.0). A real
$150 paper drawdown is impractical to stage, so drive it through the **runtime
state file** (this is *state*, not safety config — nothing in `config.py`
changes).

### 5.1 Confirm today's baseline exists
```powershell
python -X utf8 main.py live-readiness --connect   # connecting snapshots start-of-day equity
Get-Content .\logs\daily_risk_state.json
```
Note today's `start_equity` value.

### 5.2 Simulate a breaching loss (edit the STATE file only)
Set today's `start_equity` to **current equity + 200** so the bot believes the
day is already down > $150 vs. the baseline. In `logs\daily_risk_state.json`,
for today's date, raise `start_equity` accordingly (e.g. if equity ≈ 100000, set
`start_equity` to `100200`). Save.

> This is the single allowed "test helper": a **state** edit, reversible, no
> code and no config touched.

### 5.3 Observe the block
```powershell
python -X utf8 main.py paper
```
- **Audit:** a `halt` line `"phase": "entry_gate"`, `"reason": "daily_loss"`
  for any attempted entry:
  ```powershell
  Select-String -Path .\logs\order_audit.jsonl -Pattern '"reason": "daily_loss"' | Select-Object -Last 3
  ```
- **Log:** `New entry blocked for <SYM> BUY (daily_loss)`.
- **PASS:** entries blocked with `reason=daily_loss`; closes/repair still allowed.
- **Note:** as with Gate 4, the line only appears if an entry is attempted. The
  deterministic proof is `test_risk_engine` / `test_phase3_protection`; this step
  confirms the kill-switch is *wired into the live order gate*.

### 5.4 Restore the state file
Revert `start_equity` to the real value recorded in 5.1 (or delete
`logs\daily_risk_state.json`; it is regenerated on the next connect).
**PASS:** `Get-Content .\logs\daily_risk_state.json` shows the original baseline.

---

## Gate 6 — Reconnect watchdog fails closed

### 6.1 Bounded retry on initial connect, then give up (deterministic)
**With TWS / Gateway CLOSED** (or API disabled), run:
```powershell
python -X utf8 main.py paper
```
- **Observe (console):** `Could not connect to IBKR TWS …` and the process exits
  with **exit code 1** (`echo $LASTEXITCODE`).
- **Observe (audit):**
  ```powershell
  Select-String -Path .\logs\order_audit.jsonl -Pattern '"stage": "watchdog"' | Select-Object -Last 3
  ```
  → an `"event": "gave_up"`, `"healthy": false` line.
- **Observe (log):** `IBKR connect failed after 3 attempt(s) (reconnect enabled=True)`.
- **PASS:** retry is **bounded** (≤ `IBKR_RECONNECT_MAX_ATTEMPTS=3`, backoff ~2s
  then ~4s), it **terminates** (no infinite loop), and **no order is placed**.

### 6.2 Mid-run disconnect → fail closed (no new entries)
Start TWS, begin a run, then break the link mid-run (in TWS:
**API → Settings**, momentarily uncheck *Enable Socket Clients*, or use
**Global Config → API → "Reset API order ID / disconnect"**) while `paper` is
connected:
```powershell
python -X utf8 main.py paper
```
- **Observe (log):** `reconnect_watchdog: IBKR disconnected mid-run -> connection
  UNHEALTHY, failing closed (no new entries)`.
- **Observe (audit):**
  ```powershell
  Select-String -Path .\logs\order_audit.jsonl -Pattern '"event": "disconnected"' | Select-Object -Last 3
  ```
  → `"healthy": false`. Any subsequent entry audits `halt` with
  `"reason": "connection_unhealthy"`.
- **PASS:** after the drop, **no NEW entry** is placed (closes/repair are never
  routed through this gate). Because the one-shot run is short, this is timing
  sensitive; the deterministic backstop is `test_reconnect_watchdog`.

---

## Gate 7 — Graceful shutdown preserves protective stops

Already exercised in **A.2 / A.3**. Standalone confirmation: after any `paper`
run that held a protected long,
```powershell
Select-String -Path .\logs\order_audit.jsonl -Pattern '"stage": "shutdown"' | Select-Object -Last 4
```
- **PASS:** the final `shutdown` `"phase": "complete"` line shows
  `"n_orders_to_cancel": 0` and `"opens_new_entries": false`; the resting **GTC**
  stops from before the run are **still present in TWS** afterward, and the
  watchdog did **not** log a mid-run disconnect (the teardown was marked
  intentional).

---

## Gate 8 — Alerts are log-only and non-blocking

Shipped state: `ALERTS_ENABLED=False` → `alerts.emit()` is a no-op.

### 8.1 Inert-by-default (empirical)
After all runs above:
```powershell
Select-String -Path .\logs\order_audit.jsonl -Pattern '"stage": "alert"'
```
- **PASS:** **no** `alert` lines exist — alerting is inert by default and could
  not have affected any trading decision above.

### 8.2 Log-only + never-blocks (authoritative)
```powershell
python -X utf8 -m unittest test_alerts -v
```
- **PASS:** `OK`. The suite proves: disabled ⇒ no-op; enabled+log-only ⇒ logger +
  `order_audit` **only** (nothing leaves the box); `emit()` never raises and never
  places/cancels an order.

### 8.3 (Optional) See a real log-only alert, then revert
Only if you want to watch one fire. Temporarily set in `config.py`:
`ALERTS_ENABLED=True` (keep `ALERTS_LOG_ONLY=True`). Re-run a step that emits
(e.g. 6.1 reconnect give-up). Expect a log line
`ALERT [critical] reconnect_failure …` and a `"stage": "alert"` audit line, and
**no** external delivery. **Then set `ALERTS_ENABLED` back to `False`.**
- **PASS:** alert appears in log + audit only; reverted to shipped state.

---

## Final cleanup & sign-off

```powershell
python -X utf8 main.py panic-flatten            # dry-run: confirm nothing left
python -X utf8 main.py panic-flatten --confirm   # only if positions/orders remain
python -X utf8 main.py live-readiness --connect  # expect: unprotected: none; NOT READY overall
```

| Gate | What was proven | Pass evidence | Result |
|---|---|---|---|
| 1. Server-side GTC stop survives shutdown | A.2/A.3 | 2× GTC SELL (STP+TRAIL), OCA group, still resting after process exit | ☐ |
| 2. Startup reconciliation = broker truth | A.1/A.4 | `reconcile` audit `source=broker`; PASS flips with broker state | ☐ |
| 3. Unprotected long detect + repair | A.1/A.2 | `unprotected: SPY` → `Repaired protective GTC stop(s)` | ☐ |
| 4. Market-hours gate blocks entries | 4.1/4.2 | `schedule blocked reason=<...>`; `halt reason=market_closed` | ☐ |
| 5. Daily-loss kill-switch blocks entries | 5.3 | `halt reason=daily_loss` | ☐ |
| 6. Reconnect watchdog fails closed | 6.1/6.2 | `watchdog event=gave_up`; `event=disconnected healthy=false` | ☐ |
| 7. Graceful shutdown preserves stops | A.2/7 | `shutdown complete n_orders_to_cancel=0`; stops survive | ☐ |
| 8. Alerts log-only & non-blocking | 8.1/8.2 | no `alert` audit by default; `test_alerts` OK | ☐ |

**Go/No-Go:** all eight ☑ → the Phase 3/4/5 safety gates are empirically
validated on **paper**. Live conversion (Phase 6) stays **NOT READY** and out of
scope. No safety config was changed by this run-book.

---

## Appendix P — (Optional) on-demand entry-gate probe

Gates 4/5/6 only log a block when a NEW entry is actually attempted, which
depends on the model emitting a BUY. To force the entry gate **on demand** —
without waiting for a real signal and without adding any file — paste this into
an interactive `python -X utf8` session **from the project root**. It uses the
real, paper-port-locked bridge and the real gate; it places a **paper** order
only if the gate *allows* it (i.e. RTH + no loss), otherwise it returns the
block reason.

```python
import main; main._ibkr_setup_asyncio()
from ibkr_bridge import IBKRBridge
from predictor import Signal
b = IBKRBridge()
assert b.connect()                      # paper-port lock enforced here
b.snapshot_start_of_day_equity()
px = b.get_price("SPY", allow_historical=False) or 100.0
sig = Signal("SPY", "BUY", 0.99, 0.99, 0.99, 0.99, px, "gate probe")
blocked, reason = b._new_entries_blocked("SPY", 1*px, sig, px)
print("BLOCKED:", blocked, "REASON:", reason)   # e.g. market_closed / daily_loss / connection_unhealthy
b.disconnect()
```
- Outside RTH → `BLOCKED: True REASON: market_closed`.
- With the §5.2 state edit applied → `BLOCKED: True REASON: daily_loss`.
- This calls **only** the read-only gate function (`_new_entries_blocked`); it
  does **not** place an order by itself. PAPER account, port 7497 only.
