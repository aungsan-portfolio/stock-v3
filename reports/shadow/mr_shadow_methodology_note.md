# Mean Reversion Shadow Labeling Methodology & Comparability Note

## 1. Methodology Upgrade (October 1 Forward vs Sep-25 Historical Replay)
- **Forward Horizon**: 6 completed 5m bars (30-minute forward window).
- **Target**: VWAP.
- **Stop**: Close - 0.5 * ATR.

### Resolution Rule Differences
| Setup Event | Sep-25 Historical Replay | October Forward Shadow Labeler |
|---|---|---|
| Target touched before stop | REVERSION (Win) | REVERSION (Win) |
| Stop touched before target | BREAKDOWN (Loss) | BREAKDOWN (Loss) |
| Both touched in same bar | Net-change tie-break (favorable tie-break) | **AMBIGUOUS_INTRABAR (Strict Conservative Loss)** |
| Neither touched in 30 minutes | Marked-to-market at 30m | **TIME_EXPIRY_30M** (Distinct bucket, MTM PnL tracked) |

> [!WARNING]
> **Comparability Break**:
> Forward shadow breakdown rates under `label_mr_shadow_outcomes.py` are strictly more conservative than the Sep-25 replay (which found a 42.7% breakdown rate). Any forward breakdown rate must not be compared 1:1 against the 42.7% baseline without accounting for this conservative asymmetry.
