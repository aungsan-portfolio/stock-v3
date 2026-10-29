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

## 6. Development & Operational Window Policy (Outside Live Session Bug Fixes Allowed)
* **Date Ratified**: 2026-10-29
* **Operational Policy**:
  - **Code changes allowed outside live session (ICT 20:30–03:00)**: Bug fixes, bookkeeping repairs, telemetry improvements, and safety invariant hardening are authorized during off-market hours.
  - **Live session code freeze (ICT 20:30–03:00)**: Strictly zero code edits or process mutations while the live/paper market trading session is running.
  - **Strategy changes must be behind config flags**: Any new strategy logic, stop modifiers, or filter changes must be guarded by feature flags (default=True/False as specified) and logged at startup.
  - **Regression baseline**: All automated test suites must pass 100% cleanly before deployment to any live session.

### Current Operational Directives
* **Tier 2 WebSocket Defect**: ✅ **CLOSED & FROZEN**
* **Phase B Reconnect Supervisor**: ✅ **HARDENED & LOCKED**
* **Phase C Active Validation Engine**: 🟢 **ACCUMULATING $N \ge 30$ PAPER TRADES**
* **Production Trading Policy**: 🧊 **STRICTLY FROZEN (ZERO PARAMETER DRIFT)**
* **Real-Money Trading**: ⛔ **STRICTLY BLOCKED UNTIL $N \ge 30$ AUDIT**
* **Active Focus**: Measure and report exclusively on **actual fills, exits, and reconciled closed trades ($N$)** in `data/paper_validation_trades.jsonl`.

## 7. Midday Lunch Chop Filter & Empirical Governance Baseline
* **Date Ratified**: 2026-10-21
* **Policy Verdict**: **KEEP FILTER / NO PARAMETER CHANGE** (Score Threshold 60 strictly frozen)

### Core Ratified Determinations
1. **Midday Lunch Chop Filter Retained**:
   * Threshold 60 remains fully active under Phase C Runtime Code Freeze.
   * Zero parameter drift, no timeframe modifications, and no threshold relaxations permitted.
2. **Rejected Candidate Research (Observation Only)**:
   * Tracking rejected candidates remains strictly limited to telemetry and shadow evaluation.
   * Zero live/paper broker order submission; strictly zero pollution of `data/paper_validation_trades.jsonl` or production ledger.
3. **Live Promotion Boundary**:
   * Real-money live trading remains strictly BLOCKED until verified paper closed trades $N \ge 30$ and quantitative gates (PF $\ge 1.25$, Max DD $\le 5.0\%$) pass.

### Mandatory Empirical Review Protocols
1. **Hypothesis vs. Empirical Fact**:
   * Theoretical metrics (such as $E[R] = -0.2895R$ for Score < 60) are institutional risk hypotheses / benchmarks, NOT empirical facts.
   * Negative EV and counterfactual "capital saved" claims must not be asserted as verified fact without completed-trade empirical distributions.
2. **Predefined Counterfactual Analysis**:
   * Evaluation of rejected setups must strictly use pre-registered rules: signal timestamp, entry/exit criteria, forward MFE/MAE, quote spread, synthetic slippage, and fee drag.
   * Post-hoc rule modifications and look-ahead bias are strictly prohibited.
3. **Dual-Sample Independence**:
   * **Sample A (Operational Evidence)**: Reconciled paper closed trades ($N \ge 30$) for live promotion audit.
   * **Sample B (Counterfactual Research)**: Separate candidate outcome telemetry. Conflating Sample A and Sample B or using Sample B count to infer trade profitability is strictly forbidden.

## 8. Swing Sizing Deadlock & 28% Exposure Cap Baseline Audit
* **Date Ratified**: 2026-10-21
* **Policy Verdict**: **KEEP 28% GROSS EXPOSURE CAP / NO PARAMETER DRIFT / CODE FROZEN**
* **Single Source of Truth**: `MIN_ORDER_NOTIONAL = 500.00` in `strategies/swing_sizing.py`
* **Test Verification**: **54 / 54 Swing Regression Tests Passed (100% Zero Regressions)**

### Core Audit Findings
1. **Daytrade Confidence Floor (0.6000)**:
   * Confidence threshold strictly frozen at `0.6000`. Near-miss cohort ($0.58 \le \text{conf} < 0.60$, $N=190$) forward shadow analysis showed negative expectancy ($WR=26.3\%, PF=0.44, E[R]=-0.621R$).
   * No near-miss tolerance bypass permitted.
2. **Reconciled Exposure Arithmetic (2026-10-19 20:31 ICT Cycle)**:
   * Starting Net Liq Equity: **$99,769.05**
   * Stage 0 Gross Exposure Cap (28%): **$27,935.33**
   * Real Open Long Positions ($N=4$):
     - `AMZN`: 39 shares @ $256.00 = **$9,984.00**
     - `NVDA`: 45 shares @ $216.70 = **$9,751.50**
     - `MSFT`: 8 shares @ $531.64 = **$4,253.12**
     - `JPM`: 11 shares @ $338.86 = **$3,727.46**
     - Total Market Exposure: **$27,716.08** (27.78% of equity; 99.22% capacity utilized)
   * Pending BUY Orders: **$0.00** (0 pending entry orders)
   * Remaining Portfolio Budget: **$219.25** ($27,935.33 - $27,716.08)
3. **Deadlock Mechanism (`q_portfolio = 0` vs `$500` Minimum)**:
   * Higher-priced tickers (`AVGO` @ $334.78, `QQQ` @ $719.14, `GOOGL` @ $344.08):
     $$\lfloor \$219.25 / \text{price} \rfloor = 0 \rightarrow q\_portfolio = 0\text{ (ZERO\_SHARES)}$$
     Buying 1 whole share would breach the 28% cap.
   * Lower-priced ticker (`BAC` @ $51.32):
     $$\lfloor \$219.25 / 51.32 \rfloor = 4\text{ shares} \rightarrow \text{Gross Notional} = 4 \times \$51.32 = \mathbf{\$205.28}$$
     (Explains why `$205.28` was logged as gross notional).
     Since $\$205.28 < \$500.00$, it is correctly rejected by the broker safety minimum notional rule.
4. **Final Operational Directive**:
   * Production candidate is officially **FROZEN** under Phase C Runtime Code Freeze.
   * Zero feature additions, zero parameter changes, zero routine audits.
   * Focus exclusively on accumulating closed paper trades in `data/paper_validation_trades.jsonl` towards $N \ge 30$.
   * Real-money live trading remains **STRICTLY BLOCKED** until quantitative promotion gates are audited and ratified.

## 9. Swing Execution Lifecycle & Strategy Attribution Baseline Audit
* **Date Ratified**: 2026-10-21
* **Policy Verdict**: **AUDIT CLOSED WITH DOCUMENTED SCOPE LIMITS / CODE FROZEN / REAL-MONEY PROMOTION STRICTLY BLOCKED**

### Core Ratified Determinations
1. **Strategy Attribution & Performance Baseline**:
   * **Modern Swing Bot (`swg_*`)**: $N = 14$ closed trades, Win Rate = 50.0%, $PF = 0.41$, Net Realized P&L = -$928.35. Fails both Gate 1 ($N \ge 30$) and Gate 2 ($PF \ge 1.25$).
   * **Combined Swing Trades (`swg_*` + `se3-*` + other)**: $N = 40$ closed trades, Win Rate = 35.0%, $PF = 0.42$, Net Realized P&L = -$1,136.68.
   * **Daytrade Legacy Cohort (`dt_*`)**: $N = 19$ closed trades, Win Rate = 47.4%, $PF = 3.02$, Net Realized P&L = +$262.51 (historical artifact from pre-segregation testing).
   * **Total Account-Level (`PA3R71I7JFDZ`)**: $N = 59$ closed trades, Win Rate = 38.98%, $PF = 0.58$, Net Realized P&L = -$874.17.
2. **Execution Lifecycle & Sub-Penny Reconciliation**:
   * FIFO execution matching across bracket orders strictly requires sorting by economic execution timestamp (`filled_at`), not submission timestamp (`created_at`), preventing child leg tie-breaking anomalies.
   * `NVDA` partial fill VWAP: Raw fill notional = $9,756.73 across 45 shares; exact VWAP = $216.816222...; rounded display average = $216.82 (100% verified against Alpaca activities API).
   * Equity reconciliation: $\text{Cash } (\$72,055.87) + \text{Market Value } (\$28,101.82) = \text{Real-time Equity } (\$100,157.69)$. Difference from Last Close Equity ($99,997.46) is exactly $+160.23$, matching $+160.23$ in position market value change with 0 intervening cash or fill activities.
3. **Drawdown Tolerance & Documented Scope Limits**:
   * **Daily-Close Series (65 days, 1D)**: Historical Peak = $100,749.94, Trough = $99,411.04, Max Drawdown = **1.33%** ($1,338.90) $\le 5.00\%$ (**Gate 3 PASS**).
   * **7-Day Intraday Series (135 points, 15Min)**: Historical Peak = $100,173.14, Trough = $99,330.98, Max Drawdown = **0.84%** ($842.16).
   * **Documented Scope Limit**: The 7-day 15-minute series confirms short-term intraday stability but does not prove lifetime 85-day intraday drawdown. Alpaca API enforces a 30-day limit per intraday request.
4. **Final Operational Directive**:
   * Real-money live trading remains **STRICTLY BLOCKED**.
   * Phase C Runtime Code Freeze remains in full effect (zero parameter changes, 28% exposure cap, 0.6000 confidence floor, $500 minimum notional frozen).
   * Zero forced exits of open positions. Modern Swing Bot accumulates closed trades naturally towards $N \ge 30$.
