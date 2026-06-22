# Arm B tradeability evaluation (offline)

_Generated: 2026-06-22 18:51:03_

Read-only / offline. No IBKR, no orders, no model saved (rf_models.joblib / lstm_checkpoint.pt / model_metrics.json untouched).
Reuses Arm B's leakage-safe panel walk-forward CV (date-axis split, purge 5 bars = ML_HORIZON; pooled sklearn HistGradientBoosting, seed 42). Tradeable return = Close[t+5]/Close[t]-1 (H-bar hold); in binary mode this is exactly the quantity the label thresholds, so the eval is self-consistent.
Costs are deducted per trade as a TOTAL round-trip cost (net = gross - bps/1e4).

> Reproducibility: the pipeline is deterministic GIVEN the price data (offline-tested), but it pulls live yfinance auto-adjusted history, so re-running on another day — or after a dividend/split re-adjusts a symbol's history — shifts the exact figures (the AUC and high-prob tail move a little). The numbers below are a snapshot of the data as of the generation time; read the conclusions, not the last decimal.

## Setup

- Arm : **B only** (base 14 + SPY relative-strength = 18 features) | label mode : binary
- Universe : 108 requested (model_metrics.json rf keys), 108 used
- CV splits : 5 | purge : 5 | horizon : 5 bars | candlesticks : off
- Thresholds : [0.55, 0.6, 0.65, 0.7, 0.75] | top-k : [1, 3, 5, 10] | costs(bps) : [0.0, 5.0, 10.0]

## Out-of-sample prediction pool

- Rows : **104067** across 5 walk-forward folds, 2022-07-01 → 2026-06-11
- Pooled AUC : 0.5154 | base positive rate : 0.5056
- Prob [min/mean/max] : 0.0812 / 0.4729 / 0.8893
- Forward return mean / median : 0.0067 / 0.0035

## Baseline — trade every signal

_Return-space base rate: a threshold / top-k rule only adds value if it beats this at the same cost._

| all | cost(bps) | trades | hit | avg ret | med ret | PF | expect | maxDD | prec | lift | ann~ | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.0 | 104067 | 0.5324 | 0.0067 | 0.0035 | 1.3297 | 0.0067 | 118.6742 | 0.5056 | 1.0 | 0.3368 |  |
| all | 5.0 | 104067 | 0.5291 | 0.0062 | 0.003 | 1.3015 | 0.0062 | 120.5487 | 0.5056 | 1.0 | 0.3116 |  |
| all | 10.0 | 104067 | 0.5242 | 0.0057 | 0.0025 | 1.274 | 0.0057 | 122.4232 | 0.5056 | 1.0 | 0.2864 |  |

## Threshold sweep

| thr | cost(bps) | trades | hit | avg ret | med ret | PF | expect | maxDD | prec | lift | ann~ | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.55 | 0.0 | 28016 | 0.5509 | 0.0107 | 0.0055 | 1.5576 | 0.0107 | 34.827 | 0.5251 | 1.0386 | 0.539 |  |
| 0.55 | 5.0 | 28016 | 0.548 | 0.0102 | 0.005 | 1.5254 | 0.0102 | 35.5575 | 0.5251 | 1.0386 | 0.5138 |  |
| 0.55 | 10.0 | 28016 | 0.5429 | 0.0097 | 0.0045 | 1.4939 | 0.0097 | 36.288 | 0.5251 | 1.0386 | 0.4886 |  |
| 0.6 | 0.0 | 15200 | 0.5568 | 0.0118 | 0.0063 | 1.6264 | 0.0118 | 22.4466 | 0.5314 | 1.0512 | 0.5959 |  |
| 0.6 | 5.0 | 15200 | 0.5534 | 0.0113 | 0.0058 | 1.5929 | 0.0113 | 22.8626 | 0.5314 | 1.0512 | 0.5707 |  |
| 0.6 | 10.0 | 15200 | 0.5487 | 0.0108 | 0.0053 | 1.5601 | 0.0108 | 23.2786 | 0.5314 | 1.0512 | 0.5455 |  |
| 0.65 | 0.0 | 6790 | 0.5669 | 0.0177 | 0.0072 | 1.9711 | 0.0177 | 8.9145 | 0.5424 | 1.0729 | 0.8911 |  |
| 0.65 | 5.0 | 6790 | 0.5635 | 0.0172 | 0.0067 | 1.9326 | 0.0172 | 9.027 | 0.5424 | 1.0729 | 0.8659 |  |
| 0.65 | 10.0 | 6790 | 0.5585 | 0.0167 | 0.0062 | 1.8948 | 0.0167 | 9.1418 | 0.5424 | 1.0729 | 0.8407 |  |
| 0.7 | 0.0 | 2539 | 0.5888 | 0.0103 | 0.0095 | 1.6167 | 0.0103 | 6.1269 | 0.5652 | 1.1179 | 0.5166 |  |
| 0.7 | 5.0 | 2539 | 0.5865 | 0.0098 | 0.009 | 1.5795 | 0.0098 | 6.2024 | 0.5652 | 1.1179 | 0.4914 |  |
| 0.7 | 10.0 | 2539 | 0.5813 | 0.0093 | 0.0085 | 1.5431 | 0.0093 | 6.2831 | 0.5652 | 1.1179 | 0.4662 |  |
| 0.75 | 0.0 | 877 | 0.6226 | 0.0125 | 0.0128 | 1.8902 | 0.0125 | 2.6883 | 0.5975 | 1.1818 | 0.628 |  |
| 0.75 | 5.0 | 877 | 0.6214 | 0.012 | 0.0123 | 1.8431 | 0.012 | 2.7243 | 0.5975 | 1.1818 | 0.6028 |  |
| 0.75 | 10.0 | 877 | 0.6146 | 0.0115 | 0.0118 | 1.7971 | 0.0115 | 2.7603 | 0.5975 | 1.1818 | 0.5776 |  |

## Top-k per day

| k | cost(bps) | trades | hit | avg ret | med ret | PF | expect | maxDD | prec | lift | ann~ | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 990 | 0.5232 | 0.0406 | 0.0025 | 2.9952 | 0.0406 | 1.5771 | 0.497 | 0.983 | 2.0479 | outlier |
| 1 | 5.0 | 990 | 0.5212 | 0.0401 | 0.002 | 2.9478 | 0.0401 | 1.5986 | 0.497 | 0.983 | 2.0227 | outlier |
| 1 | 10.0 | 990 | 0.5162 | 0.0396 | 0.0015 | 2.9013 | 0.0396 | 1.6347 | 0.497 | 0.983 | 1.9975 | outlier |
| 3 | 0.0 | 2970 | 0.5286 | 0.0254 | 0.0032 | 2.2387 | 0.0254 | 4.4165 | 0.5024 | 0.9937 | 1.2827 | outlier |
| 3 | 5.0 | 2970 | 0.5259 | 0.0249 | 0.0027 | 2.2006 | 0.0249 | 4.598 | 0.5024 | 0.9937 | 1.2575 | outlier |
| 3 | 10.0 | 2970 | 0.5215 | 0.0244 | 0.0022 | 2.1632 | 0.0244 | 4.7795 | 0.5024 | 0.9937 | 1.2323 | outlier |
| 5 | 0.0 | 4950 | 0.5339 | 0.0177 | 0.0037 | 1.878 | 0.0177 | 6.3897 | 0.5071 | 1.003 | 0.8911 | outlier |
| 5 | 5.0 | 4950 | 0.5313 | 0.0172 | 0.0032 | 1.8434 | 0.0172 | 6.4859 | 0.5071 | 1.003 | 0.8659 | outlier |
| 5 | 10.0 | 4950 | 0.5265 | 0.0167 | 0.0027 | 1.8095 | 0.0167 | 6.5919 | 0.5071 | 1.003 | 0.8407 | outlier |
| 10 | 0.0 | 9900 | 0.534 | 0.0153 | 0.0039 | 1.7426 | 0.0153 | 11.7703 | 0.509 | 1.0068 | 0.7703 |  |
| 10 | 5.0 | 9900 | 0.5313 | 0.0148 | 0.0034 | 1.7103 | 0.0148 | 11.9253 | 0.509 | 1.0068 | 0.7451 | outlier |
| 10 | 10.0 | 9900 | 0.5265 | 0.0143 | 0.0029 | 1.6786 | 0.0143 | 12.0803 | 0.509 | 1.0068 | 0.7199 | outlier |

**Verdict — SELECTION EDGE: 14/18 costed selections beat the all-signal baseline on median return AND precision-lift; best by median is threshold=0.75 @ 5.0bps: median 0.0123 (baseline 0.003), mean 0.012 (baseline 0.0062), hit 0.6214, PF 1.8431, lift 1.1818, 877 trades. Weak but centered — worth a guarded, still-offline follow-up (robustness across regimes, position overlap, capacity at the top-prob names). NOT a go-live signal.**

_Legend: PF=profit factor (n/a = no losing trades), expect=per-trade expectancy (mean net return), maxDD=max drawdown of the cumulative per-trade equity curve, prec=precision vs label, lift=precision/base-rate, ann~=rough non-compounded annualized (avg×252/horizon, ignores overlap), flag='outlier' when mean expectancy > 4x |median| (a few fat-tailed winners, not a centered edge — treat the mean with suspicion and read the median). Full per-row OOS predictions are in the .json under `oos_predictions` (columnar)._