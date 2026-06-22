# Pooled cross-symbol GBM A/B (offline edge test)

_Generated: 2026-06-22 13:07:28_

Read-only / offline ML evaluation. No IBKR, no orders, no model saved (rf_models.joblib / lstm_checkpoint.pt / model_metrics.json untouched).
Leakage-safe panel walk-forward CV: split on the global DATE axis with a purge gap of 5 bars (ML_HORIZON); pooled GBM = sklearn HistGradientBoostingClassifier.

## Setup

- Universe : 108 symbols requested (model_metrics.json rf keys)
- Label modes : ['binary'] | arms : ['A', 'B']
- CV splits : 5 | purge : 5 | null draws : 100 (scope within_symbol)
- Candlestick features : off | seed : 42

### Per-symbol RF baseline (model_metrics.json (binary-trained RF))

- Mean AUC **0.5068** | median 0.5072 | symbols with AUC: 8/108

## binary — arm A (14 features, 108 symbols, 125521 rows)

- **Pooled CV-AUC : 0.4983** (null mean 0.5028, p95 0.506, p-value **0.9901**, 100 draws)
- Precision@BUY(0.65) : 0.503 on 1662 signals | positive rate 0.4944
- Per-symbol AUC : mean 0.5028 median 0.4988 [0.4362..0.7052], 5/108 > 0.55

**Verdict — NO EDGE: pooled AUC 0.4983 (p=0.9901) vs RF baseline 0.5068. Pooling does not recover edge from this feature/label set.**

## binary — arm B (18 features, 108 symbols, 124009 rows)

- **Pooled CV-AUC : 0.5162** (null mean 0.5038, p95 0.5068, p-value **0.0099**, 100 draws)
- Precision@BUY(0.65) : 0.5475 on 6795 signals | positive rate 0.4964
- Per-symbol AUC : mean 0.5169 median 0.5167 [0.4532..0.5755], 3/108 > 0.55

**Verdict — NO EDGE: pooled AUC 0.5162 (p=0.0099) vs RF baseline 0.5068. Pooling does not recover edge from this feature/label set.**
