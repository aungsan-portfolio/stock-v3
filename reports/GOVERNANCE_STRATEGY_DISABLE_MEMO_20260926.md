# Governance Action Memo: Reversible Disable of MOMENTUM_SCALP and VWAP_BOUNCE

**Date**: 2026-09-26  
**Auditor**: Forensic Analysis Engine  
**Authority**: Aung San (Principal)  
**Status**: RATIFIED & ACTIVE  

```text
=== GOVERNANCE PROVENANCE ===
Proposed by : Antigravity (Forensic Auditor)
Approved by : Aung San (Principal)
Date        : 2026-09-26
Scope       : Loss Containment / Execution Policy
Type        : Reversible Config Flag Disable (Zero Code Deletion)
```

---

## 1. Executive Summary & Rationale

Following the comprehensive forensic audit of production trade journals, paper validation ledgers, and gate ledgers on 2026-09-26, two intraday trading strategies exhibited catastrophic performance degradation with severe negative expectancy:

1. **`MOMENTUM_SCALP`**:
   - **Combined Sample**: $N = 32$ closed trades (Paper: 19, Production: 13)
   - **Profit Factor**: $PF = 0.28$ (Production $PF = 0.00$)
   - **Win Rate**: $WR = 12.5\%$ (Production $WR = 7.7\%$)
   - **Cumulative Net Loss**: $-\$472.45$
   - **Verdict**: Immediate loss containment required.

2. **`VWAP_BOUNCE`**:
   - **Combined Sample**: $N = 65$ closed trades (Paper: 55, Production: 10)
   - **Profit Factor**: $PF \approx 0.05$ (Paper $PF = 0.33$, Production $PF = 0.00$)
   - **Win Rate**: $WR \approx 10.8\%$ (Production $WR = 10.0\%$)
   - **Cumulative Net Loss**: $-\$158.00$
   - **Era Caveat**: September degradation coincided with the cluster cap pollution era and sector concentration false blocks, causing execution drag on bounce setups.
   - **Verdict**: Immediate loss containment required.


---

## 2. Technical Implementation Architecture

To preserve statistical integrity and funnel transparency while containing capital drain:

1. **Reversible Configuration Flags (`config.py`)**:
   - `MOMENTUM_SCALP_ENABLED: bool = False` (aliased to `STRATEGY_MOMENTUM_SCALP_ENABLED = False`)
   - `VWAP_BOUNCE_ENABLED: bool = False` (aliased to `STRATEGY_VWAP_BOUNCE_ENABLED = False`)
   - Reversible in 5 seconds via standard configuration toggles.
   - Strategy code and underlying models remain 100% intact.

2. **Funnel Denominator Preservation (`strategies/orchestrator.py`)**:
   - Strategies continue to execute their evaluation routines against incoming market data.
   - Generated candidate signals are intercepted at `STAGE_REGIME` before broker submission.
   - Signals are logged to `data/trade_gate_decision.jsonl` with:
     - `decision = "REJECT"`
     - `reason_code = VetoReason.STRATEGY_DISABLED`
     - `human_message = "Strategy disabled by configuration (governance loss containment)"`
   - Gating funnel denominators remain complete and accurate for statistical review.

---

## 3. Forward-Only Re-entry Criteria (Promotion Back to Active)

Neither strategy may be re-enabled for paper execution or real-money consideration without satisfying the following quantitative promotion gates:

1. **Forward Sample Size**: $N \ge 20$ newly generated, forward-only paper signals.
2. **Gross Profit Factor**: $PF \ge 1.00$.
3. **No Historical Pollution**: Re-entry evaluation must use purely clean forward data collected in `data/paper_validation_trades.jsonl`; historical or backfilled records are strictly barred from re-entry calculations.
4. **Governance Approval**: Re-enablement requires an explicit, written sign-off memo by Aung San (Principal).
