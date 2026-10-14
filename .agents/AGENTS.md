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

### Binding Hard Gates (Evaluated for Live Stage 1 Admission)
1. **Statistical Validity & Duration (Gate 1)**: $N \ge 30$ closed trades AND $\ge 2$ calendar weeks elapsed ($\ge 14$ days).
2. **Sample Diversity (Gate 1b)**: $\ge 10$ distinct trading days AND single-day contribution $\le 30\%$ of total $N$ (max 9 trades on any single day).
3. **Profit Factor (Gate 2)**: $PF \ge 1.25$.
4. **Drawdown Tolerance (Gate 3)**: $\text{Max Drawdown} \le 5.0\%$ of starting equity (strictly half of the 10% emergency halt breaker).

### Reported Diagnostic Co-Metrics (Informative / Non-Blocking)
* **Win Rate**: Target reference $\ge 40.0\%$ (informative co-metric confirming R:R asymmetry).
* **Execution Slippage**: Target reference $\le 0.05\%$ (broker-simulated approximate fill footnote).

### Two-Stage Real-Money Promotion Architecture (Risk Scaling Layer)
* **Stage 1 (N ≥ 30 Hard Gates Pass)**: Admission to Live Trading at **REDUCED sizing** (25% – 50% target risk). Protects capital against $N=30$ statistical sample noise (95% CI on Win Rate $\approx \pm 14-18\%$).
* **Stage 2 (N ≥ 80–100 Sustained)**: Scaled to **FULL real-money sizing** only after $N \ge 80-100$ trades maintain $PF \ge 1.25$ with zero gate regression.

### Dual-Base Reconciled Portfolio Heat Statement
* **Paper Testing Baseline ($10,000.00 Equity Base)**:
  - Daily Loss Circuit Breaker: **$300.00** (`MAX_DAILY_LOSS_DAYTRADE`).
  - Maximum Daily Equity Heat: **3.0%** of capital ($300 / $10,000).
  - Per-Trade Risk Cap: $50.00 (`MAX_RISK_PER_TRADE_DOLLARS`), representing **0.5%** of capital (or $100 / 1.0% if unconstrained). Breaker halts trading at cumulative **-6R** (or **-3R** at 1.0%).
  - Gate 3 Drawdown Ceiling (5.0%): strictly **$500.00** on $10,000 starting equity.
* **Production / Scaled Live Baseline ($100,000.00 Equity Base)**:
  - Daily Loss Circuit Breaker: **$300.00** (`MAX_DAILY_LOSS_DAYTRADE`).
  - Maximum Daily Equity Heat: **0.3%** of capital ($300 / $100,000).
  - Per-Trade Risk: 0.1% equity ($100.00 on $100,000 capital via `DAYTRADE_PAPER_RISK_PCT = 0.001`). Breaker halts trading at cumulative **-3R**.
  - Maximum Open Exposure: `MAX_OPEN_POSITIONS = 7` ($700.00 total concurrent open risk). If 3 consecutive positions hit stop, the breaker triggers `FLATTEN_ON_DAILY_LOSS`, capping portfolio heat at 0.3%.
  - Gate 3 Drawdown Ceiling (5.0%): strictly **$5,000.00** on $100,000 starting equity.

## 4. TRANSITIONING Texture Filter — Research Findings
* **Date**: 2026-08-31
* **Sample**: $N = 322$ rejected signals (walk-forward 5m candle simulation)

| Metric | Empirical Value | Gate Threshold | Gate Status |
|---|---|---|---|
| **Win Rate** | **32.3%** | $\ge 40\%$ (Policy Gate Floor) | ❌ FAILED |
| **Gross Profit Factor** | **1.10** | $\ge 1.25$ | ❌ FAILED |
| **Reward / Risk** | **2.29 : 1** | $\ge 1.50 : 1$ | ✅ PASS |
| **Avg Win / Avg Loss** | **+0.309% / -0.135%** | — | — |
| **Net Expectancy** | **-0.0408% / trade** | $> 0.0\%$ | ❌ NEGATIVE |
| **Cumulative Expected Drag** | **-13.1% (Linear: 322 * -0.0408%)** | — | Informative Linear Drag |
| **True Portfolio Max Drawdown** | **NOT COMPUTED / NOT VERIFIED** | $\le 5.0\%$ | ⚠️ NOT EVALUABLE |

> [!IMPORTANT]
> **Decision Rule**: The `TRANSITIONING` texture filter is **empirically justified by negative Net Expectancy (-4.08 bps/trade) and PF < 1.25**, NOT by a verified portfolio drawdown breach.
> **Cost Model Attribution**: Implied cost drag is $4.92\text{ bps/trade}$ ($+0.0084\%$ gross vs $-0.0408\%$ net), representing broker friction model (slippage, spread half-cost, and fees). Break-even win rate without friction is $30.4\%$; the $40.0\%$ threshold serves as an institutional safety margin.
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
* **Swing Engine Canonical Status**: Option C shadow running; production models remain on verified Aug-29 vintage (`models/lstm_checkpoint.pt`, `models/rf_models.joblib`); gate audit scheduled for November 6; real-money trading strictly BLOCKED without passing quantitative promotion gates.

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


