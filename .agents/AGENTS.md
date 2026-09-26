# Workspace Customization Rules

This file defines the project-scoped operational constraints and rules for stock engine trading activities.

## 1. Trader Location & Timezone
* **Location**: Chiang Mai, Thailand (ICT, UTC+7 timezone).
* **Daylight Saving Time (DST / EDT)** (Summer/Fall):
  - Pre-market scanner: **8:00 PM - 8:15 PM** Thailand Time.
  - Regular Market trading: **8:30 PM - 3:00 AM (next day)** Thailand Time.
* **Standard Time (EST)** (Winter/Spring):
  - Pre-market scanner: **9:00 PM - 9:15 PM** Thailand Time.
  - Regular Market trading: **9:30 PM - 4:00 AM (next day)** Thailand Time.

## 2. Multi-Account Execution Safety (CRITICAL)
> [!IMPORTANT]
> **DO NOT** run Day Trading (`daytrade-bot`) and Swing Trading (`paper`/`predict`) on the same single Alpaca account.
> 
> **Why**: Day Trading contains automated end-of-day flatten rules (`flatten_all()`) and risk circuit breakers. When triggered, these API calls will wipe out all resting orders and close *every* position on the account, accidentally liquidating long-term Swing trading positions.
> 
> **Rule**: Always use separate Alpaca Sub-accounts (with separate API keys) to keep Day Trading and Swing Trading assets completely isolated.

## 3. Quantitative Promotion Gates (Paper -> Real Money)
Before migrating the Day Trading strategy from paper trading to real money trading, the following thresholds must be met (audited via `evaluate_paper_results.py`):
1. **Statistical Validity**: $N \ge 30$ closed trades.
2. **Profit Factor**: $PF \ge 1.25$.
3. **Win Rate**: $WR \ge 40\%$.
4. **Drawdown Tolerance**: $\text{Max Drawdown} \le 5\%$ (calculated on a trade-by-trade intraday basis).
5. **Slippage Audit**: $\text{Avg Slippage} \le 0.05\%$.

## 4. TRANSITIONING Texture Filter — Research Findings
* **Date**: 2026-08-31
* **Sample**: $N = 322$ rejected signals (walk-forward 5m candle simulation)

| Metric | Empirical Value | Gate Threshold | Gate Status |
|---|---|---|---|
| **Win Rate** | **32.3%** | $\ge 40\%$ | ❌ FAILED |
| **Gross Profit Factor** | **1.10** | $\ge 1.25$ | ❌ FAILED |
| **Reward / Risk** | **2.29 : 1** | $\ge 1.50 : 1$ | ✅ PASS |
| **Avg Win / Avg Loss** | **+0.309% / -0.135%** | — | — |
| **Net Expectancy** | **-0.0408% / trade** | $> 0.0\%$ | ❌ NEGATIVE |
| **Expected Drawdown** | **-13.1% (over 322 trades)** | $\le 5.0\%$ | ❌ VIOLATION |

> [!IMPORTANT]
> **Decision Rule**: The `TRANSITIONING` texture filter is **mathematically justified and locked**.
> **DO NOT** allow `MOMENTUM_SCALP` or `CANDLESTICK` in `TRANSITIONING` texture.
> **Next Review Gate**: Only re-evaluate when $N \ge 500$ new sample signals are recorded or under a verified market regime transition.

## 5. Cross-Engine Exposure Governance — RATIFIED (Option A: Full Strategy Segregation)

**Policy**: Option A — Full Strategy Segregation

### Core Invariant
Daytrade and Swing engines operate in separate broker sub-accounts.
Each engine owns and controls only its assigned broker account.

Concurrent opposite-direction positions in the same underlying symbol across the two sub-accounts are **permitted** and are **not automatically vetoed**.

Example:
* Swing account (`PA3R71I7JFDZ`): Long BAC
* Daytrade account (`PA3UODIC5JTA`): Short BAC

This is classified as **cross-engine tactical exposure**, not as an execution-control violation.

### Required Safety Boundaries
1. **Broker Account Isolation — Mandatory**:
   Each engine must remain bound to its configured broker account.
   Cross-engine emergency actions must never operate on the other engine's account.

2. **No Cross-Account Order Mutation**:
   An engine may not:
   * cancel another engine's orders;
   * flatten another engine's positions;
   * modify another engine's protective orders;
   * infer ownership from symbol alone.
   Account identity remains part of the execution-control boundary.

3. **Aggregate Exposure — Telemetry Only**:
   Opposing exposure for the same underlying across engines must be observable in diagnostics and session reporting.
   Required telemetry should identify, at minimum:
   `symbol`, `engine`, `account`, `side`, `quantity`, `timestamp`.
   The existence of opposing exposure does **not** itself authorize or block an order.

4. **Independent Strategy Risk Controls Remain Authoritative**:
   Each engine continues to enforce its own:
   * regime gates;
   * confidence gates;
   * position sizing;
   * daily loss limits;
   * protection requirements;
   * execution controls;
   * broker reconciliation rules.
   Cross-engine awareness must not silently override these engine-local policies.

5. **Promotion Boundary**:
   This policy does **not** remove any other real-money promotion blocker.
   Market-data provenance, protection/recovery certification, rotation validation, and all other explicitly locked promotion gates remain independently enforceable.

### Forbidden Policy Drift
The following are prohibited without a separately ratified governance amendment:
* automatic opposing-position veto;
* hidden cross-engine position netting;
* cross-account flattening;
* cross-engine sizing modification;
* cross-engine strategy threshold modification.

### Final Governance Status
* **CROSS-ENGINE EXPOSURE POLICY**: OPTION A — RATIFIED
* **Opposing same-symbol exposure across segregated sub-accounts**: PERMITTED
* **Cross-account execution control**: PROHIBITED
* **Aggregate exposure visibility**: REQUIRED
* **Strategy-policy independence**: REQUIRED
* **Real-money promotion blockers**: UNCHANGED

## 6. Runtime Code Freeze & Phase C Empirical Sample Collection Gate
* **Date Ratified**: 2026-09-25
* **Verification Baseline**: **76 / 76 Automated Tests Passed (100% Zero Regressions)**
* **Operational Phase**: **Fixing Phase is Officially CLOSED**. The system is transitioned to **Paper Runtime Sample Collection ($N \ge 30$)**.

### Strict Code Freeze Protocol
1. **No Routine Code Modifications**:
   Normal runtime events (e.g. routine WebSocket reconnect backoffs, soft gate rejections, lunch chop skips, or zero-trade market sessions) are **NOT defects** and must **NOT** trigger code changes.
2. **Observe + Record Only**:
   Telemetry, warnings, and gate rejections must be observed and recorded without modifying strategy thresholds or operational code.
3. **Critical Defect Re-Opening Criteria**:
   Code modifications (Tier 1 / Tier 2) are **strictly forbidden** unless one of the following 6 critical invariants is breached with empirical evidence:
   * **Breach 1: Trading safety violation** (fail-closed invariant broken, orders placed during stream disconnection).
   * **Breach 2: Broker truth discrepancy** (unreconciled state or desynchronization with Alpaca API).
   * **Breach 3: Protection failure** (missing, dropped, or cancelled stop-loss on open positions).
   * **Breach 4: Idempotency failure** (duplicate order submission or client order ID collision).
   * **Breach 5: Accounting leak** (validation lane orders or P&L polluting production ledger).
   * **Breach 6: Invariant regression** (failure of any of the 76 locked regression test cases).

### Current Operational Directives
* **Tier 2 WebSocket Defect**: ✅ **CLOSED & FROZEN**
* **Phase B Reconnect Supervisor**: ✅ **HARDENED & LOCKED**
* **Phase C Active Validation Engine**: 🟢 **ACCUMULATING $N \ge 30$ PAPER TRADES**
* **Production Trading Policy**: 🧊 **STRICTLY FROZEN (ZERO PARAMETER DRIFT)**
* **Real-Money Trading**: ⛔ **STRICTLY BLOCKED UNTIL $N \ge 30$ AUDIT**
* **Active Focus**: Measure and report exclusively on **actual fills, exits, and reconciled closed trades ($N$)** in `data/paper_validation_trades.jsonl`.


