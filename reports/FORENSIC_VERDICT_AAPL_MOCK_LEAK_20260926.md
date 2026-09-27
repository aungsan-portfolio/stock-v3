# Forensic Verdict Record: AAPL Mock-Leak Quarantine

**Date**: 2026-09-26  
**Auditor**: Forensic Analysis Engine  
**Status**: APPROVED & LOCKED  

```text
=== VERDICT RECORD: AAPL MOCK-LEAK QUARANTINE (2026-09-26) ===
Subject : trade_journal_production_clean.jsonl lines 730, 824
          AAPL 100sh $150->$140, pnl -$1,000 each, "Emergency flatten exit"
Verdict : SYNTHETIC TEST LEAKAGE — QUARANTINED (flag-only, reversible,
          raw untouched; source SHA-256 18f476e23d490b6b9aa9f6fa5d427688d3056c1ab658654dac5f4357a4d6f982 verified post-run)

Evidence:
  V1 broker truth : logs/broker_events.jsonl AAPL fills (Sep) = qty 1.0 @
                    $339.76 / $339.97 / $340.23 only. ZERO 100sh/$150/$140.
                    [evidence class: LOCAL event log, not direct API query —
                     direct Alpaca API cross-check = optional addendum]
  V2 provenance   : (a) 2026-09-05 = Saturday — live fill impossible
                    (b) mock contract verbatim in tests/
                        test_swing_execution_safety.py:863,1178
                        (AAPL qty=100, avg=150, stop=140)
                    (c) notes string emitted at trailing_stop.py:1623
                    (d) verbatim duplicates in trade_journal_archive_jul22
  V3 gate ledger  : ledger starts 2026-09-17 -> 09-05/09-09 dates are
                    PRE-LEDGER (absence there is not independent proof).
                    Positive proof: only AAPL PASS anywhere = 09-21, qty=11,
                    cid=se3-AAPL-B-mome-20260921-e43e0986b88aaa71.
                    No 100-share AAPL PASS exists.
  V4 risk snapshot: swing_position_risk.json — AAPL absent

Precision notes:
  - 09-05 record: Saturday alone kills live-fill possibility outright.
  - 09-09 record (Wednesday): rests on V1 + V2(d) + fingerprint, not calendar.
  - Quarantine fingerprint requires ALL 6 mock fields to match (reversible).

OV-1 RESOLVED (2026-09-26): Dual-path test-data egress confirmed.
  (1) 2026-09-05 16:03 UTC record: tests/test_swing_execution_safety.py
      invoked log_trade without journal_file override -> default production
      path (Saturday, no market dependency in journaling).
  (2) 2026-09-09 12:37 UTC record: originated from second direct test execution on
      2026-09-09 (write-time stamp), persisted through reconstruct->reconcile
      passthrough (commit 109bee2) which lacked lane flagging; is_validation=True
      filter admitted unflagged mock records into production clean journal.
  Preventive coverage: Patch 1 tripwire, Patch 2 lane guard, Patch 3 provenance.
OPEN ITEM OV-2: 25 legacy UNKNOWNs (Jul-Aug) permanently unattributable —
  documented, accepted.

Addendum (2026-09-26 Post-Run Verification & Governance Sign-off):
  1. Archive Recovery Caveat: The 6 trades recovered via `archive_join_20260722`
     were re-tagged from UNKNOWN to RECONCILED with cryptographic provenance markers.
     Named strategy attribution was not assigned because the source archive itself
     used "RECONCILED" as a placeholder (proving attribution degradation existed
     in the July archive pipeline).
  2. Historical Regime vs. September Bleed:
     - July–August Legacy Era: Profitable (+$59.29, PF 1.40, WR 32.0%).
     - September Era: Sharp degradation driven by regime transition, execution friction,
       and sector cluster cap false blocks.
  3. Master Audit Metrics (N=64 active closed trades, post-quarantine):
     - UNKNOWN (July/Aug): n=25, WR=32.0%, PF=1.40, PnL=+$59.29
     - RECONCILED (10 native + 6 archive): n=16, WR=31.2%, PF=0.47, PnL=-$167.19
     - MOMENTUM_SCALP: n=13, WR=7.7%, PF=0.00, PnL=-$467.15
     - VWAP_BOUNCE: n=10, WR=10.0%, PF=0.00, PnL=-$142.36
     - TOTAL: n=64, WR=23.4%, PF=0.33, Net PnL=-$717.41, Exp=-$11.21
  4. Real-Money Trading Gate: STRICTLY BLOCKED until Phase C N>=30 clean forward audit passes.
     Additional promotion thresholds (PF>=1.25, WR>=40%, Exp>=0) are PROPOSED, NOT RATIFIED
     (principal sign-off required).
  5. Governance Provenance:
     - Proposed by: Antigravity (Forensic Auditor)
     - Approved by: Aung San (Principal)
     - Date: 2026-09-26
  6. Architectural Amendment Status & Amendment A Ratification:
     - Amendment A (Winner Preservation / Threshold): RATIFIED AS SUPERSEDED BY DESIGN (Reading 2).
       * Adjudication: Winner protection is fully satisfied upstream in candidate selection (`position_rotator.py:444` `is_profit_accelerating` skip). Downstream execution in `atomic_rotation.py` remains non-blocking audit logging.
       * Invariant Lock: Regression test added to `tests/test_position_rotator.py` (`test_profit_accelerating_winner_never_rotation_candidate`).
       * Backlog Disposition: Reading 1 is explicitly dismissed and excluded from post-freeze re-evaluation backlog.
       * Freeze Discipline: Test-only addition; zero production code modification.
     - Amendment B (Rotator DI): P2 testability refactor logged (call-time DI verified in bridge, non-blocking).
     - Amendment C (ReplacementTruthStatus): VERIFIED & FROZEN (29/29 tests passed in test_atomic_rotation.py).
     - Amendment D (Cancel-then-Close): VERIFIED & FROZEN (6/6 tests passed in test_swing_rotation_hardening.py, 3.0s convergence fail-closed).
     - Amendment E (Journal Isolation Tripwires): APPLIED & LOCKED.
     - Amendment F (Recovered_via Contract): LIVE CONTRACT (recovered_via enum contract live; flatten_reconciled_fallback isolated to journal-metadata namespace).
```

## Canonical System Memory (Rev 2 — RATIFIED BY PRINCIPAL 2026-09-27)

```text
# CANONICAL SYSTEM MEMORY (Rev 2 — RATIFIED BY PRINCIPAL 2026-09-27)
TRUTH BASELINE: source journal production_clean.jsonl (1,231 rec, 66 closed,
  SHA 18f476e2..., untouched); recovered artifact = trade_journal_attribution_
  recovered.jsonl (SHA f31c1069...); quarantine = in-place reversible flags
  (2 AAPL mocks, -$2,000), NO separate quarantine file.
LOCKED METRICS (post-quarantine): N=64, PF=0.33, WR=23.4%, Exp=-$11.21, Net=-$717.41.
LEGACY ERA: Jul-Aug UNKNOWN batch net +$59.29 (PF 1.40); bleed era = Sep regime friction.
DISABLED (reversible): MOMENTUM_SCALP, VWAP_BOUNCE (memos 2026-09-26);
  re-entry forward-only N>=20 PF>=1.0 + principal sign-off; backfilled data excluded.
RECOVERED_VIA registry: client_order_id_parse | gate_order_id_join |
  gate_session_tiebreak | archive_join_20260722; journal-metadata namespace:
  flatten_reconciled_fallback; native records = field ABSENT.
AMENDMENTS: A = RATIFIED SUPERSEDED BY DESIGN (winner protection satisfied
  upstream via position_rotator.py is_profit_accelerating skip; locked via
  test_position_rotator.py; Reading 1 excluded from backlog);
  B = P2 refactor logged; C = verified 29/29; D = verified 6/6 fail-closed;
  E = applied; F = recovered_via contract live.
OV-1: CLOSED (dual-path egress, Line B, commit 109bee2 context).
PHASE C: N=0/30 via tools/phase_c_session_report.py;
  STRATEGY_DISABLED veto first-occurrence session = wiring verification event.
REAL MONEY: BLOCKED until Phase C N>=30 clean forward audit passes.
  Additional promotion thresholds (PF>=1.25 / WR>=40% / Exp>=0) = PROPOSED,
  NOT RATIFIED — principal sign-off required.
FREEZE: Rule 6 maintained; 6 breach classes documented.
```


