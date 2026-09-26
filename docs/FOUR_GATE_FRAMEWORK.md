# Four-Gate Framework — Consolidated Production Gate Standard

This document establishes the official quantitative and infrastructural gating requirements for the Stock Engine Pro Day Trading Bot. It defines the progression from paper trading to real-money deployment.

Sequence: Gate 1 -> Gate 2 -> Gate 3 -> Gate 4
Principle: Evidence-gated, NOT calendar-gated.

---

## Gate 1: Deterministic Infrastructure

```
+------------------------------------------------------+
|  GATE 1: DETERMINISTIC INFRASTRUCTURE                |
+------------------------------------------------------+
|                                                      |
|  Latency Decomposition (T0->T6 per event):           |
|  [ ] T0: Exchange event timestamp (SIP feed)         |
|  [ ] T1: Bar available to strategy                   |
|  [ ] T2: Signal generated (bar close eval)           |
|  [ ] T3: Order submitted                             |
|  [ ] T4: Broker ACK                                  |
|  [ ] T5: First partial fill                          |
|  [ ] T6: Full fill                                   |
|                                                      |
|  Measurable from log:                                |
|  [ ] Compute latency  T3-T2  target: < 100ms         |
|  [ ] Broker ACK       T4-T3  p95   < 500ms           |
|  [ ] Fill latency     T5-T4  tracked separately:     |
|      Marketable fill p95 < 2s                        |
|      Limit queue wait classified separately          |
|                                                      |
|  Currently unmeasured (must be instrumented):        |
|  [ ] Data latency T1-T0 (feed age at strategy)       |
|  [ ] 285s limit wait root cause unattributed:        |
|      Option A: Limit price $759.14 not reached       |
|      Option B: IEX feed stale, signal stale          |
|      Option C: Combined A + B                        |
|                                                      |
|  Reliability:                                        |
|  [ ] WS reconnect includes mandatory reconciliation  |
|  [ ] Broker positions match internal state post-recon|
|  [ ] Stale signals auto-expire (strategy-specific age|
|  [ ] Clock sync verified (exchange vs local)         |
|  [ ] Feed source recorded per bar event              |
|                                                      |
|  Reporting:                                          |
|  [ ] All latency metrics reported as p50/p95/p99/max |
|  [ ] Average latency alone explicitly rejected       |
|                                                      |
|  EXIT CRITERIA:                                      |
|  [ ] All T0-T6 timestamps instrumented               |
|  [ ] Data age (T1-T0) measured and within threshold  |
|  [ ] 285s fill ambiguity root-cause resolved         |
|  [ ] SIP feed confirmed active (not IEX only)        |
|  [ ] All boxes above checked                         |
+------------------------------------------------------+
```

---

## Gate 2: Statistical Strategy Validation

```
+------------------------------------------------------+
|  GATE 2: STATISTICAL STRATEGY VALIDATION             |
+------------------------------------------------------+
|                                                      |
|  Minimum Counts (all must be met simultaneously):    |
|  [ ] Sessions:            >= 30-50                   |
|  [ ] Closed trades:       >= 100                     |
|                                                      |
|  Statistical basis for >=100 trade requirement:      |
|  margin of error +-10% at 95% confidence:            |
|    p=0.60 -> n=92   p=0.75 -> n=72                   |
|    Conservative floor: 100 (not arbitrary)           |
|                                                      |
|  Regime Coverage (all buckets required):             |
|  [ ] RISK_ON / TRENDING:   >= 20 trades              |
|  [ ] RISK_ON / SIDEWAYS:   >= 20 trades              |
|  [ ] RISK_OFF or mixed:    >= 10 trades              |
|  Note: Session 1 = RISK_ON/SIDEWAYS only             |
|        Cannot generalize from single regime          |
|                                                      |
|  Performance Thresholds (after estimated costs):     |
|  [ ] Win Rate:             >= 40% (with CI reported) |
|  [ ] Profit Factor:        >= 1.25                   |
|  [ ] Expectancy:           > 0 per trade             |
|  [ ] Max Drawdown:         from equity curve         |
|                          (not aggregate budget)      |
|                                                      |
|  Stability Checks:                                   |
|  [ ] No single symbol   > 40% of total P&L           |
|  [ ] No single strategy > 60% of total trades        |
|  [ ] Performance stable across all regime buckets    |
|                                                      |
|  Near-Miss Dataset:                                  |
|  [ ] Confidence 0.585-0.600 signals collected        |
|  [ ] MFE / MAE / Expected R recorded per bucket      |
|  [ ] Signal families deduplicated (not inflated)     |
|                                                      |
|  Signal Half-Life Study (per strategy type):         |
|  [ ] Forward return decay measured empirically       |
|  [ ] Half-life = t where E[R(t)] <= 0.5 * E[R(t=0)]  |
|  [ ] Max signal age gate set individually:           |
|      MEAN_REVERSION  = empirical value (TBD)         |
|      VWAP_BOUNCE     = empirical value (TBD)         |
|      MOMENTUM        = empirical value (TBD)         |
|  [ ] Single shared threshold explicitly rejected     |
|  [ ] Conservative default until empirical:           |
|      reject signals > 3 bar ages old                 |
|                                                      |
|  Semantic Note:                                      |
|  "Strategy logic executed as designed" = PASS        |
|  "Strategy has positive expectancy"    = NOT YET     |
|   Requires N>=100 closed trades across               |
|   diverse regime conditions                          |
|                                                      |
|  EXIT CRITERIA:                                      |
|  [ ] All count minimums met                          |
|  [ ] All regime buckets filled                       |
|  [ ] Confidence intervals reported (not point est.)  |
|  [ ] Half-life empirically established per strategy  |
+------------------------------------------------------+
```

---

## Gate 3: Micro-Live Canary

```
+------------------------------------------------------+
|  GATE 3: MICRO-LIVE CANARY                           |
+------------------------------------------------------+
|                                                      |
|  Purpose:                                            |
|  Verify live broker behavior differs from paper      |
|  at minimum financial risk before full sizing.       |
|  P&L alone cannot authorize stage advancement.       |
|                                                      |
|  Sizing Progression (each stage >= 20 closed trades):|
|  [ ] Stage 1: 1-share minimum                        |
|     -> verify order routing and fill confirmation    |
|  [ ] Stage 2: 10% risk ($5 max per trade)            |
|     -> verify slippage and actual spread cost        |
|  [ ] Stage 3: 25% risk ($12.50 max per trade)        |
|     -> verify partial-fill handling                  |
|  [ ] Stage 4: 50% risk ($25 max per trade)           |
|     -> verify position and stop management at scale  |
|  [ ] Stage 5: Full approved risk ($50 max per trade)  |
|     -> verify full system behavior                   |
|                                                      |
|  Minimum Evidence Per Stage (all required):          |
|  [ ] >= 20 closed live trades                        |
|  [ ] >= 5 distinct symbols                           |
|  [ ] Both target and protective exits observed       |
|  [ ] >= 1 partial-fill lifecycle observed            |
|  [ ] >= 1 reconnect reconciliation tested live         |
|  [ ] No unresolved position or order mismatch        |
|                                                      |
|  Live Execution Metrics:                             |
|  [ ] Market-data age at signal time recorded         |
|  [ ] Order ACK latency: p50/p95/p99/max              |
|  [ ] Marketable-fill latency: separately measured     |
|  [ ] Limit-order queue duration: separately measured  |
|  [ ] Implementation shortfall decomposed             |
|  [ ] Spread and regulatory costs recorded             |
|  [ ] Cancel/replace success rate reported            |
|  [ ] Fill ratio and partial-fill ratio reported       |
|                                                      |
|  State Integrity Checks:                             |
|  [ ] Broker positions match internal positions        |
|  [ ] Broker open orders match internal open orders    |
|  [ ] Parent and protective orders remain linked       |
|  [ ] Duplicate exposure remains zero                  |
|  [ ] Reconciliation runs after every reconnect        |
|  [ ] End-of-day flat-book check passes                |
|                                                      |
|  Safety Controls:                                    |
|  [ ] Kill switch tested with real broker orders       |
|  [ ] Daily loss guard tested under live equity        |
|  [ ] Position-size ceiling independently enforced     |
|  [ ] Maximum open-position count enforced             |
|  [ ] Stale signals rejected before submission         |
|  [ ] Orphan orders detected and cancelled             |
|  [ ] Manual emergency liquidation procedure tested    |
|                                                      |
|  EXIT CRITERIA:                                      |
|  [ ] Stage 5 completed                                |
|  [ ] No critical state failure at any stage           |
|  [ ] Modeled costs validated against actual costs     |
|  [ ] All risk guards verified and operational         |
|  [ ] Operational sign-off recorded                    |
+------------------------------------------------------+
```

### Gate 3: Decision Logic

```python
canary_decision_rules = {

    "hard_stop": [
        "broker_internal_position_mismatch",
        "duplicate_order_or_duplicate_position",
        "protective_stop_missing",
        "kill_switch_unresponsive",
        "daily_loss_limit_breached_without_shutdown",
        "unknown_orphan_order",
        "unreconciled_state_after_reconnect",
        "position_exceeds_approved_risk",
    ],

    "pause_and_review": [
        "slippage_p95_above_approved_threshold",
        "marketable_fill_latency_p95_above_target",
        "fill_ratio_below_80_percent",
        "cancel_replace_failure",
        "actual_cost_above_3x_modeled_cost",
        "clock_drift_above_tolerance",
        "unexpected_partial_fill_pattern",
    ],

    "advance_to_next_stage": [
        "minimum_20_closed_trades",
        "minimum_symbol_diversity_met",
        "no_hard_stop_event",
        "all_orders_reconciled",
        "protective_orders_verified",
        "actual_costs_within_modeled_range",
        "manual_review_completed",
    ],
}
```

### Hard Stop Recovery Sequence

```
Hard Stop triggered:
  1. New entries immediately disabled
  2. Open positions reconciled against broker
  3. Protective orders verified present
  4. Unsafe exposure flattened if necessary
  5. All orphan orders cancelled
  6. Incident snapshot preserved
  7. Root-cause analysis completed
  8. Fix deployed to paper environment
  9. Regression test passed
 10. Canary resumes at same or lower stage
     (failure severity determines which stage)

Critical rule:
  Automatic advancement after critical failure
  is never permitted.
```

### Implementation Shortfall Decomposition

```
Decision Price = Price when strategy decided to trade
Arrival Price  = Market price when order reached broker
Execution Price = VWAP actual fill price

For BUY orders:

decision_shortfall_bps = (
    (execution_price - decision_price)
    / decision_price * 10_000
)

routing_shortfall_bps = (
    (execution_price - arrival_price)
    / arrival_price * 10_000
)

delay_cost_bps = (
    (arrival_price - decision_price)
    / decision_price * 10_000
)

Where the loss occurs:

Decision -> Arrival   = data age, compute delay, or
                       submission delay
Arrival  -> Execution = spread, venue routing, queue,
                       or liquidity cost
Execution -> Exit     = strategy behavior and
                       trade management
```

### Paper-to-Live Comparison Report

```
Per live signal, record in parallel:
  Live:  actual order, actual fill
  Paper: hypothetical fill using identical signal

Compare per matched pair:
  [ ] Fill probability
  [ ] Average fill price (bps difference)
  [ ] Partial-fill behavior
  [ ] Time to fill
  [ ] Slippage
  [ ] Exit timing
  [ ] Net P&L after actual costs

Promotion report must include:
  Live minus Paper delta for:
  [ ] Fill-rate difference
  [ ] Entry-price difference (bps)
  [ ] Exit-price difference (bps)
  [ ] Holding-time difference
  [ ] Gross expectancy difference
  [ ] Net expectancy difference
  [ ] Profit-factor difference
  [ ] Stop-trigger behavior difference

Interpretation rule:
  Large paper-to-live gap != strategy malfunction
  First audit: execution model, queue assumptions,
  transaction cost model - before blaming alpha.
```

---

## Gate 4: Enhancement Promotion

```
+------------------------------------------------------+
|  GATE 4: ENHANCEMENT PROMOTION                       |
+------------------------------------------------------+
|                                                      |
|  Purpose:                                            |
|  Test new alpha features without contaminating       |
|  or destabilizing the validated core system.         |
|                                                      |
|  Experimental Features (current):                    |
|  [ ] Experiment 1: Sector Exception Layer            |
|  [ ] Experiment 2: Near-Threshold Confidence Sizing  |
|  [ ] Experiment N: Future regime/execution features  |
|                                                      |
|  Isolation Requirements:                             |
|  [ ] Each feature has an independent feature flag    |
|  [ ] Core strategy is always the control group       |
|  [ ] Every signal carries feature-attribution tags   |
|  [ ] Experimental risk tracked separately            |
|  [ ] Experimental P&L tracked separately             |
|  [ ] Core thresholds remain unchanged                |
|  [ ] Immediate rollback supported at all times       |
|  [ ] Feature disabled = core bot continues running   |
|                                                      |
|  Phase 1: Offline Evaluation                         |
|  [ ] Historical replay completed                     |
|  [ ] Look-ahead bias audit completed                 |
|  [ ] Transaction costs included in results           |
|  [ ] Parameter sensitivity tested                    |
|  [ ] Regime-level breakdown reported                 |
|                                                      |
|  Phase 2: Shadow Mode                                |
|  [ ] Signals generated, no orders submitted          |
|  [ ] Entry and exit prices simulated prospectively   |
|  [ ] MFE, MAE, Expected R captured per signal        |
|  [ ] >= 20 sector-exception candidates collected     |
|  [ ] Near-threshold dataset reaches required count   |
|                                                      |
|  Phase 3: A/B Paper Test                             |
|  [ ] Control and treatment use same data stream      |
|  [ ] Core and experimental P&L attributed separately |
|  [ ] >= 2 weeks of prospective testing               |
|  [ ] No degradation of global risk metrics           |
|                                                      |
|  Phase 4: Micro-Live Feature Canary                  |
|  [ ] Minimum size only                               |
|  [ ] Separate daily risk budget                      |
|  [ ] Automatic rollback conditions active            |
|  [ ] Core system runs independently with flag off    |
|                                                      |
|  EXIT CRITERIA:                                      |
|  [ ] Incremental expectancy positive after costs     |
|  [ ] Risk metrics do not regress vs core baseline    |
|  [ ] Attribution is clean and auditable              |
|  [ ] Rollback verified functional                    |
+------------------------------------------------------+
```

### Experiment 1: Sector Exception Layer

```python
# Candidate qualification (additive permission,
# not a broad-regime override)
sector_exception_candidate = (
    broad_texture == "SIDEWAYS"
    and sector_divergence >= configured_threshold
    and sector_breadth   >= configured_min_breadth
    and symbol in approved_sector_leaders
    and liquidity_check_passed
    and spread_check_passed
)

# Core risk gates still apply - no bypass:
# [ ] Maximum risk per trade
# [ ] Noise-floor validation
# [ ] Chase guard
# [ ] Liquidity filter
# [ ] Spread filter
# [ ] Daily loss guard
# [ ] Maximum correlated exposure
# [ ] Stale-signal guard

# Correlated exposure control
# (position count alone is insufficient)
# NVDA + AMD + AVGO = 3 symbols, 1 factor exposure
sector_risk = sum(
    p.initial_risk
    for p in open_positions
    if p.sector == target_sector
)
allow_new_trade = (
    sector_risk + candidate.initial_risk
    <= max_sector_risk_budget
)

# Shadow record per candidate
sector_shadow_record = {
    "feature":            "SECTOR_EXCEPTION",
    "feature_version":    "v1",
    "symbol":             None,
    "broad_regime":       None,
    "sector":             None,
    "sector_divergence":  None,
    "sector_breadth":     None,
    "signal_time":        None,
    "signal_price":       None,
    "hypothetical_entry": None,
    "mfe":                None,
    "mae":                None,
    "exit_price":         None,
    "expected_r":         None,
    "net_r_after_costs":  None,
}
```

### Experiment 2: Near-Threshold Confidence Sizing

```python
# Signal family deduplication (critical):
# QQQ same side/strategy cycling 0.5989->0.5981
# over 5 bars = 1 independent setup, not 5.

signal_family_key = (
    symbol,
    side,
    strategy,
    session_date,
    setup_origin_time,
)

# Counting rule:
# Raw evaluations      = 5
# Independent setups   = 1
# Potential trades     = 1
# (inflate to 5 = sample inflation error)

# Sizing rule:
# Control:   confidence >= 0.600 -> normal risk
# Treatment: 0.585 <= confidence < 0.600
#            -> shadow mode first
#            -> reduced size only after validation

# Stop rule (important):
# Preferred initial experiment:
#   Same structurally valid stop
#   + Reduced position size
#   + Same target framework
#
# NOT automatically:
#   Reduced position size
#   + Arbitrarily tighter stop
#   (tighter stop increases noise-floor violations)
```

### Near-Threshold Promotion Criteria

```
[ ] Independent setup count sufficient (not raw evals)
[ ] Positive Expected R after transaction costs
[ ] MAE does not materially exceed control group
[ ] MFE capture remains economically meaningful
[ ] Performance persists outside QQQ
[ ] Performance persists across multiple sessions
[ ] Results not driven by single regime
[ ] Reduced sizing improves portfolio-level expectancy
```

### Feature Rollback Rules

```python
feature_rollback_conditions = {

    "sector_exception": [
        "incremental_expectancy_below_zero",
        "sector_drawdown_above_limit",
        "correlated_exposure_limit_breached",
        "core_risk_metric_regression",
        "feature_attribution_failure",
    ],

    "near_threshold_sizing": [
        "expected_r_after_costs_below_zero",
        "mae_materially_worse_than_control",
        "profit_factor_below_required_floor",
        "sample_concentration_in_single_symbol",
        "duplicate_signal_family_counting_detected",
    ],
}
```

---

## Final Promotion Matrix

```
+------------------------------------------------------+
|  CONSOLIDATED PROMOTION STATUS                       |
+------------------------------------------------------+
|                                                      |
|  Software Functionality                              |
|  Status : PASS                                       |
|  Basis  : Observed under paper conditions            |
|                                                      |
|  Risk and Order-State Behavior                       |
|  Status : VALIDATED                                  |
|  Basis  : Observed state transitions confirmed       |
|                                                      |
|  Infrastructure Determinism                          |
|  Status : PENDING                                    |
|  Reason : T0/T1 data age not instrumented            |
|           285s limit wait unattributed               |
|                                                      |
|  Statistical Strategy Edge                           |
|  Status : NOT YET PROVEN                             |
|  Reason : N < 100 closed trades                      |
|           Regime coverage incomplete                 |
|           (RISK_ON/SIDEWAYS only so far)             |
|                                                      |
|  Normal-Size Live Deployment                         |
|  Status : NOT APPROVED                               |
|                                                      |
|  Micro-Live Canary                                   |
|  Status : CONDITIONALLY ELIGIBLE                     |
|  Prereq : Gate 1 AND Gate 2 must pass first          |
|                                                      |
|  Sector Exception Layer                              |
|  Status : SHADOW MODE ONLY                           |
|                                                      |
|  Near-Threshold Sizing                               |
|  Status : DATA COLLECTION ONLY                       |
|                                                      |
|  Current Maturity Classification                     |
|  LEVEL 4 RESILIENT -- LEVEL 5 CANDIDATE              |
+------------------------------------------------------+
```

---

## Final Synthesis

> The bot has demonstrated **operational resilience** and **correct state behavior** under the paper-trading conditions observed so far. **Statistical expectancy, deterministic latency attribution, and live execution fidelity remain unproven.** Normal-size live deployment should therefore remain blocked until infrastructure and statistical gates pass, followed by a staged micro-live canary.

| Category | Status |
|---|---|
| State behavior & risk rules | Validated |
| Software correctness | Pass (paper conditions) |
| Data age (T0->T1) | Pending instrumentation |
| Fill latency root cause | Pending attribution |
| Positive expectancy | Not yet proven |
| Live execution costs | Unmeasured |
| Cross-regime stability | Single regime only |
| Full-size live deployment | Blocked |

**Next concrete actions:**

1. Instrument T0/T1 timestamps per bar event
2. Resolve 285s fill ambiguity (limit price vs feed staleness)
3. Continue paper telemetry - target 30 sessions, 100 closed trades
4. Accumulate regime diversity (TRENDING and RISK_OFF buckets)
5. Sector exception shadow mode running in parallel
