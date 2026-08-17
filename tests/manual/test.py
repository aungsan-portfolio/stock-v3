"""
test.py — Smoke tests. Runs without IBKR. Safe to invoke after `train`.
"""
from data_manager import fetch_ohlcv, build_features, make_labels
from predictor import Predictor
import config


def main() -> None:
    print("=== Step 1: Fetch Data ===")
    try:
        df = fetch_ohlcv("SPY")
        print(f"OK rows: {len(df)}")
    except Exception as e:
        print(f"FAIL: {e}")
        return

    print("\n=== Step 2: Build Features + Labels ===")
    feat = build_features(df)
    labels = make_labels(feat)
    print(f"features={len(feat)}  labels(non-nan)={int(labels.notna().sum())}")

    print("\n=== Step 3: Check Models On Disk ===")
    print(f"RF model:   {config.ML_MODELS_FILE.exists()}")
    print(f"LSTM model: {config.LSTM_CKPT_FILE.exists()}")

    print("\n=== Step 4: Predict ===")
    try:
        signals = Predictor().predict_all()
        print(f"Signals: {len(signals)}")
        for s in signals:
            print(f"  {s.symbol:<6} {s.action:<5} conf={s.confidence:.2f} ${s.price:.2f}")
    except Exception as e:
        print(f"FAIL: {e}")


if __name__ == "__main__":
    main()
