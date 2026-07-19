"""
lstm_hyperopt.py — Optuna-based hyperparameter tuning for LSTM.

Sweeps:
- LSTM_HIDDEN [32, 64, 128]
- LSTM_LAYERS [1, 2, 3]
- LSTM_LR [1e-4, 1e-3, 5e-3]
- LSTM_BIDIRECTIONAL [True, False]
- LSTM_DROPOUT [0.1, 0.3, 0.5]
- LSTM_CYCLE_LR_STEP [0, 5, 10]
- ML_HORIZON (label boundary) [3, 5, 10]

Usage:
  pip install optuna --break-system-packages
  python -X utf8 lstm_hyperopt.py --symbol SPY --trials 20
"""
import argparse
import sys
import logging
import copy
import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)

def check_optuna():
    try:
        import optuna
        return optuna
    except ImportError:
        print("Optuna is not installed. To run hyperparameter optimization, run:")
        print("  pip install optuna --break-system-packages")
        return None

def run_optimization(symbol: str, n_trials: int = 15):
    optuna = check_optuna()
    if not optuna:
        return 1

    # Disable noisy PyTorch/Optuna logs during search
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("lstm_engine").setLevel(logging.WARNING)

    from data_manager import fetch_ohlcv, build_features, make_labels, get_feature_columns
    from lstm_engine import StockLSTMEngine

    print(f"=== Tuning hyperparameters for {symbol} (trials={n_trials}) ===")

    df = fetch_ohlcv(symbol)
    engine = StockLSTMEngine()

    # Capture original configs to restore later
    orig_hidden = config.LSTM_HIDDEN
    orig_layers = config.LSTM_LAYERS
    orig_lr = config.LSTM_LR
    orig_dropout = config.LSTM_DROPOUT
    orig_bidirectional = getattr(config, "LSTM_BIDIRECTIONAL", False)
    orig_cycle = getattr(config, "LSTM_CYCLE_LR_STEP", 0)

    def objective(trial):
        # Sample parameters
        config.LSTM_HIDDEN = trial.suggest_categorical("LSTM_HIDDEN", [32, 64, 128])
        config.LSTM_LAYERS = trial.suggest_int("LSTM_LAYERS", 1, 3)
        config.LSTM_LR = trial.suggest_float("LSTM_LR", 1e-4, 5e-3, log=True)
        config.LSTM_DROPOUT = trial.suggest_float("LSTM_DROPOUT", 0.1, 0.5)
        config.LSTM_BIDIRECTIONAL = trial.suggest_categorical("LSTM_BIDIRECTIONAL", [True, False])
        config.LSTM_CYCLE_LR_STEP = trial.suggest_categorical("LSTM_CYCLE_LR_STEP", [0, 5, 10])

        try:
            # Train and return validation loss
            res = engine.train([symbol], verbose=False)
            if symbol not in res:
                return float("inf")
            # Objective: minimize validation loss, or maximize validation AUC (returned as negative)
            auc = res[symbol].get("best_val_auc")
            val_loss = res[symbol].get("best_val_loss", float("inf"))
            if auc is not None and auc > 0:
                # We prioritize AUC but also regularize with validation loss
                return -float(auc) + 0.1 * float(val_loss)
            return float(val_loss)
        except Exception:
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("\n✓ Optimization complete!")
    print("Best trial params:")
    for k, v in study.best_params.items():
        print(f"  {k:<20} : {v}")
    print(f"Best objective score : {study.best_value:.4f}")

    # Restore original config
    config.LSTM_HIDDEN = orig_hidden
    config.LSTM_LAYERS = orig_layers
    config.LSTM_LR = orig_lr
    config.LSTM_DROPOUT = orig_dropout
    config.LSTM_BIDIRECTIONAL = orig_bidirectional
    config.LSTM_CYCLE_LR_STEP = orig_cycle
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tuning LSTM hyperparameters via Optuna")
    parser.add_argument("--symbol", type=str, default="SPY", help="Symbol to tune")
    parser.add_argument("--trials", type=int, default=10, help="Number of search trials")
    args = parser.parse_args()

    sys.exit(run_optimization(args.symbol, args.trials))
