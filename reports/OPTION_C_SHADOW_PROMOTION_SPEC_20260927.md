# Specification: Option C Shadow Promotion Protocol & Forward Gate Framework

**Date Ratified**: 2026-09-27  
**Governance Provenance**:
- **Defect Flagging**: Aung San (Principal) — Identified gate logic bug, CV fold=0 false-pass, and duration mismatch.
- **Diagnostic Quantification**: Antigravity (Forensic Agent) — Identified in-sample leakage mechanism and quantified Lift = 0.84 degradation.
- **Protocol Formulation**: Joint Pair-Programming Alignment (Option C Adopted).
**Domain**: Swing Prediction Engine (`ai_engine.py`, `lstm_engine.py`, `predictor.py`)  
**Scope**: Per-Symbol Shadow Tracking & October 2026 Out-of-Sample Forward Promotion Gate.

---

## 1. Shadow Prediction Log Schema (`data/shadow_predictions.jsonl`)

The shadow tracking pipeline operates **append-only**.
To eliminate lookahead bias, `realized_outcome` fields MUST remain `null` at prediction time and are backfilled only when the forward 5-bar horizon matures.

### JSONL Record Schema
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "record_type": "SHADOW_INFERENCE_PAIR",
  "prediction_id": "pred_20261001_AAPL_093000",
  "timestamp": "2026-10-01T13:30:00Z",
  "symbol": "AAPL",
  "bar_date": "2026-10-01",
  "bar_close": 228.50,
  "incumbent": {
    "vintage": "2026-08-29",
    "rf_prob": 0.5210,
    "lstm_prob": 0.5400,
    "ensemble_conf": 0.5305,
    "action": "HOLD"
  },
  "shadow_v20260927": {
    "vintage": "2026-09-27",
    "rf_prob": 0.5820,
    "lstm_prob": 0.6100,
    "ensemble_conf": 0.5960,
    "action": "BUY"
  },
  "realized_outcome": {
    "evaluated": false,
    "eval_timestamp": null,
    "horizon_bars": 5,
    "future_bar_date": null,
    "future_close": null,
    "realized_return": null,
    "binary_label": null
  }
}
```

### Invariants:
1. **Append-Only at Bar Close**: Inference records are appended at the close of each trading day during the daily scan/prediction run.
2. **Strict Anti-Lookahead**: `realized_outcome.binary_label` is strictly `null` when the inference is written. Backfill occurs $t+5$ bars later.
3. **Execution Isolation**: The shadow ensemble output is logged for observation only; zero broker orders are placed from shadow signals.

---

## 2. Load-Path Isolation Verification (Airtight)

Codebase grep across `ai_engine.py`, `lstm_engine.py`, and `predictor.py` confirms **zero directory globbing (`glob`, `os.listdir`, `iterdir`)**.

- **RF Production Loader** (`ai_engine.py:304`):
  ```python
  if not self.cfg.ML_MODELS_FILE.exists():
      return False
  loaded = joblib.load(self.cfg.ML_MODELS_FILE)
  ```
  Strictly binds to `models/rf_models.joblib`.
- **LSTM Production Loader** (`lstm_engine.py:540`):
  ```python
  if not self.cfg.LSTM_CKPT_FILE.exists() or not self.scalers_file.exists():
      return False
  ckpt = torch.load(self.cfg.LSTM_CKPT_FILE, ...)
  ```
  Strictly binds to `models/lstm_checkpoint.pt`.
- **Shadow Isolation**:
  The shadow directory `models/shadow_v20260927/` is located outside the active configuration paths and cannot be loaded by production callers unless explicitly pointed to by an offline evaluation tool.

---

## 3. October Out-of-Sample Forward Gate Protocol

### Evaluation Rules:
1. **Evaluation Window**: Trading days in October 2026.
2. **Maturity Threshold**: $N \ge 15$ realized bars with matured 5-bar forward labels per symbol.
3. **Fair Parallel Comparison**: Both incumbent and shadow models are evaluated on the exact same October bar subset.
4. **Per-Symbol Decision Granularity**: Promotion is decided strictly per-symbol. There is no pooled "all-or-nothing" blanket promotion.

### Per-Symbol Quantitative Gate Thresholds:
| Gate | Metric | Formula / Criterion | Minimum Passing Threshold | Rationale |
|---|---|---|---|---|
| **Gate 1** | **Predictive Lift** | $\text{Lift} = \frac{\text{Precision@Buy}(\ge 0.55)}{\text{Base Positive Rate}}$ | $\ge \mathbf{1.00}$ | Model must select winners better than random coin-toss. |
| **Gate 2** | **Ranking Quality** | $\text{ROC-AUC}$ on forward October bars | $\ge \mathbf{0.5500}$ | Model must demonstrate positive discriminative ranking. |
| **Gate 3** | **Calibration Guard** | $\text{Brier Score (Shadow)} \le \text{Brier Score (Incumbent)}$ | $\le \mathbf{Brier}_{\text{incumbent}}$ | Model calibration and probability confidence must not degrade. |
| **Fallback** | **Sample Sufficiency** | Realized observations $N$ | $\ge \mathbf{15\text{ bars}}$ | $N < 15$ auto-holds incumbent (insufficient sample). |

---

## 4. Per-Symbol Promotion Sign-Off Template

When an evaluation window matures, a promotion memo must be drafted for each qualifying candidate using this exact format:

```markdown
# Model Promotion Sign-off Memo: [SYMBOL]

**Date**: YYYY-MM-DD  
**Symbol**: [SYMBOL]  
**Evaluator**: Forensic ML Supervisor  
**Approver**: Aung San (Principal)  

### 1. October Forward Performance Audit (N = [XX] bars)
| Metric | Incumbent (Aug-29) | Shadow (Sep-27) | Threshold | Status |
|---|---|---|---|---|
| Base Rate | [XX.X]% | [XX.X]% | — | — |
| Precision@Buy (>= 0.55) | [XX.X]% | [XX.X]% | — | — |
| Lift | [X.XX] | [X.XX] | >= 1.00 | [PASS / FAIL] |
| ROC-AUC | [0.XXXX] | [0.XXXX] | >= 0.5500 | [PASS / FAIL] |
| Brier Score | [0.XXXX] | [0.XXXX] | <= Incumbent | [PASS / FAIL] |

### 2. Failure Mode Review
- False-positive trap verified absent: [YES / NO]
- High-confidence distribution stability verified: [YES / NO]

### 3. Promotion Action
- [ ] PROMOTE shadow model to production disk (`models/rf_models.joblib`, `models/lstm_checkpoint.pt`)
- [ ] RETAIN incumbent model (Gate failed or sample insufficient)

**Principal Signature**: _______________________  
**Timestamp**: _________________________________
```

---

## 5. Custody & Safety Rules

1. **Pre-Evaluation Integrity Check**:
   Before running October forward evaluation, the cryptographic SHA-256 hashes of both incumbent and shadow models must be re-verified against the locked manifest:
   - Incumbent RF: `FDE80BAAF0A27A49112189AFA8D1568D9DBFFE4EE1278CD4F0D2C229CDF4EFE7`
   - Incumbent LSTM: `65CD66EEBBF31690E9FC050B5DB1F0C8154E371DF9E5533DAE2DEA9A233A52C0`
   - Shadow RF: `614CC1F3BFD89E533F2F5EE65D88017C70AF5399D431540A939E36CF7559395C`
   - Shadow LSTM: `5D13F896174F10ED131A3D023DE23C5CB2352200D5A44D9AE05A1FCDE5150A6D`
2. **Zero Auto-Promotion**: No automated script or cron may overwrite production model files. Promotion requires explicit principal sign-off.
3. **Atomic Swap & Vintage Preservation**:
   Upon promotion of any symbol, existing production model files are archived to:
   `models/backups/vintage_YYYYMMDD_HHMMSS/`
   before performing the in-place atomic update.
4. **Lane Orthogonality Invariant**:
   Swing model tracking and promotion remain strictly isolated from Day Trading execution and the Phase C Daytrade validation lane (`data/paper_validation_trades.jsonl`).
