# Forensic Diagnosis Record: ML Model Staleness, Degradation Audit & Option C Protocol

**Date**: 2026-09-27  
**Auditor**: Forensic Analysis Engine & Principal (Pair-Programming Review)  
**Domain**: Swing Prediction Engine (`ai_engine.py`, `lstm_engine.py`, `predictor.py`)  
**Status**: INCUMBENT PRESERVED (Option C Shadow Framework Adopted)

```text
=== FORENSIC RECORD: ML STALENESS & IN-SAMPLE LEAKAGE AUDIT (2026-09-27) ===
Subject : models/rf_models.joblib, models/lstm_checkpoint.pt (Aug 29 vintage vs Sep 27 retrain)
Status  : Staged in Shadow Mode. Incumbent models restored to production. Zero in-sample promotion.

1. Cryptographic Model Custody Manifest:
   - Incumbent RF   (Active Production): SHA-256 FDE80BAAF0A27A49112189AFA8D1568D9DBFFE4EE1278CD4F0D2C229CDF4EFE7
   - Incumbent LSTM (Active Production): SHA-256 65CD66EEBBF31690E9FC050B5DB1F0C8154E371DF9E5533DAE2DEA9A233A52C0
   - Shadow RF      (models/shadow_v20260927/): SHA-256 614CC1F3BFD89E533F2F5EE65D88017C70AF5399D431540A939E36CF7559395C
   - Shadow LSTM    (models/shadow_v20260927/): SHA-256 5D13F896174F10ED131A3D023DE23C5CB2352200D5A44D9AE05A1FCDE5150A6D
   - Rollback Source: models/backups/snapshot_20260927_pre_diagnostic/

2. Lane Orthogonality Audit (Execution Boundary Confirmed):
   - Grep verification across `strategies/*.py`: ZERO dependencies on `ai_engine`, `StockRFEngine`,
     `StockLSTMEngine`, or `ML_MODELS_FILE`.
   - Daytrade orchestrator `DAYTRADE_ML_ENABLED = False`. (References `SignalPredictor` at
     `models/signal_predictor.pkl`, an entirely separate model file).
   - VERDICT: Swing ML models operate in 100% orthogonal execution isolation. Retraining,
     diagnosing, or modifying Swing models has ZERO interaction or reset requirement on the
     Phase C Daytrade validation lane (`data/paper_validation_trades.jsonl`).

3. Empirical Degradation Findings (September 2026 Forward Eval):
   - Eval Window: 2026-09-01 to 2026-09-26 (13 realized bars per symbol across 20 Watchlist symbols; N=260).
   - Stale Model Pooled AUC: 0.4396 (sub-random).
   - Headline Degradation Metric:
       Pooled Precision@Buy (>= 0.55) = 0.3786 vs Base Rate 0.4500 -> Lift = 0.84 (< 1.0).
       The 0.55 threshold actively selected worse than coin-toss base rate.
   - Severe Failure Modes:
       * AMZN: 11 buy signals issued, only 1 win (precision 9.1%).
       * NFLX: precision 0.0% (n=3), AUC 0.2333.
       * QQQ : precision 0.0% (n=2), AUC 0.3500.
       * XLV : precision None (n=0), AUC 0.2500.
   - Statistical Caveats:
       * Individual symbol sample size (n=13) carries wide 95% CI (+/- 0.25); symbol-level
         distinctions are noisy.
       * META (100% positive) and UNH (100% negative) reflect one-class degenerate horizon artifacts.
       * Macro conclusion rests on the pooled lift (0.84) and AMZN false-positive trap.

4. In-Sample Promotion Leakage Trap Identified:
   - Defect Identified: Retraining models on data through Sep 26 and then testing them against
     the Sep 01-26 window constitutes in-sample evaluation.
   - The observed metric boost (AUC 0.8163, Prec 78.7%) was an artifact of in-sample fit.
   - Governance Action: Promoting models based on in-sample data is structurally invalid.
     Production models were immediately rolled back to the pristine Aug-29 incumbent files.
     The Sep-27 retrained models were isolated to `models/shadow_v20260927/`.

5. Adopted Policy: Option C (Shadow Mode + Out-of-Sample Forward Gate):
   - Architecture:
       * Step 1: All 20 watchlist symbols retrained in a single run (clean single vintage).
       * Step 2: Retrained vintage isolated in shadow mode (`models/shadow_v20260927/`).
                 Live signal path retains incumbent models; zero production risk.
       * Step 3: Forward Gate Window: Minimum 15-20 out-of-sample forward trading bars in October 2026.
       * Step 4: Per-Symbol Promotion Criteria:
                 (a) Lift >= 1.0 (Precision@Buy >= Base Rate)
                 (b) ROC-AUC >= 0.55
                 (c) Calibration Error (Brier) <= Incumbent Brier
       * Step 5: Promotion occurs on a per-symbol basis only upon passing the forward gate,
                 requiring explicit principal sign-off. Failing symbols retain incumbent.
   - Parameter Backlog:
       * Per-symbol threshold calibration (e.g. AMZN threshold tuning) deferred to post-freeze backlog.
```
