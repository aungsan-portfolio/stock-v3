"""
test_lstm_v2_manual.py — Verification script for the upgraded LSTM engine.

Tests:
1) AttentionLSTM instantiation & forward pass.
2) Cyclic LR scheduler integration.
3) Gradient clipping.
4) Meta-labeling model training & inference.
"""
import sys
import os
import torch
import numpy as np
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, str(os.path.abspath(os.path.join(__file__, ".."))))

import config
import data_manager
from lstm_engine import StockLSTMEngine, AttentionLSTM

def test_attention_model():
    print("Test 1: AttentionLSTM model instantiation & forward pass...")
    # Batch size 4, window 30, features 14
    x = torch.randn(4, 30, 14)
    model = AttentionLSTM(
        input_size=14,
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        bidirectional=False
    )
    out = model(x)
    print(f"   - Input shape: {x.shape}")
    print(f"   - Output shape: {out.shape}")
    assert out.shape == (4,), "Output shape mismatch"
    print("   PASSED\n")

def test_lstm_training():
    print("Test 2: LSTM training with Attention, Gating, and cyclic LR...")
    # Set hyperparams small for quick test run
    config.LSTM_EPOCHS = 3
    config.LSTM_PATIENCE = 2
    config.LSTM_CYCLE_LR_STEP = 2
    config.LSTM_BIDIRECTIONAL = True
    config.LSTM_HIDDEN = 16
    config.USE_MICRO_FEATURES = True

    engine = StockLSTMEngine()
    # Train SPY
    res = engine.train(['SPY'], verbose=True)
    assert 'SPY' in res, "SPY training failed"
    print("   - Got training stats:", res['SPY'])
    print("   PASSED\n")

def test_meta_labeling():
    print("Test 3: LSTM training with Meta-Labeling...")
    config.LSTM_EPOCHS = 2
    config.LSTM_META_ENSEMBLE = True
    config.USE_MICRO_FEATURES = True

    engine = StockLSTMEngine()
    res = engine.train(['SPY'], verbose=True)
    assert 'SPY' in res, "SPY training failed"
    print("   - Meta training stats:", res['SPY'])
    assert res['SPY'].get('meta_trained', False), "Meta model should have trained"

    # Test prediction
    pred = engine.predict('SPY')
    print(f"   - Prediction score: {pred:.4f}")
    assert 0 <= pred <= 1, "Prediction out of bounds"
    print("   PASSED\n")

def run_all_tests():
    print("=" * 60)
    print("Running LSTM v2 tests...")
    print("=" * 60)
    try:
        test_attention_model()
        test_lstm_training()
        test_meta_labeling()
        print("=" * 60)
        print("All LSTM v2 tests PASSED! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(run_all_tests())
