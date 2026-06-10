"""
errors.py — Shared exception types.

Kept in a dependency-free module so both engines and the predictor can import it
without creating an import cycle (predictor imports the engines).
"""


class ModelNotAvailableError(Exception):
    """Raised when a trained model/scaler for a specific symbol is unavailable.

    The predictor treats this as a *missing* sub-model (ok=False) rather than a
    neutral 0.5 score. This keeps the MIN_ML_MODELS_FOR_SIGNAL safety guard
    honest: a symbol with no trained model cannot silently count as an available
    ML model and unlock technical-only trading.
    """
