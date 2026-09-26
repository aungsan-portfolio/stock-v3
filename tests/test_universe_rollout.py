"""
tests/test_universe_rollout.py — Contract tests for Phase 1 Universe Rollout & Shadow Boundary.

Validates:
- T3-A: CVX BUY signal zero broker mutation (rotation, execute_all, execute_signal, _place_open_bracket)
- T3-B: CVX SELL signal zero broker mutation (no order cancellation, no order submission)
- T3-C: CVX HOLD signal zero broker mutation (no order cancellation, no order submission)
- T4:   CVX sector mapping in config.SECTOR_MAPPINGS ("ENERGY", "STATIC_CONFIG")
- Invariants:
    * Active WATCHLIST count == 20 (CAT, LLY, CVX not in active watchlist)
    * SHADOW_MONITOR contains {"CAT", "LLY", "CVX"}
    * DAYTRADE_WATCHLIST count == 17, USE_DYNAMIC_SCANNER == False
    * Sector cap == 0.25, cluster risk == 0.0125
"""
import pytest
from unittest.mock import MagicMock, patch

import config
from alpaca_bridge import AlpacaBridge
from predictor import Signal


@pytest.fixture
def mock_bridge():
    """Create a mock AlpacaBridge in swing mode with a fully instrumented client."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge._client.get_account.return_value = MagicMock(
        account_number="PA3R71I7JFDZ", equity=100_000.0, buying_power=100_000.0
    )
    bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=200.0))
    bridge.get_net_liquidation = MagicMock(return_value=100_000.0)
    bridge._client.get_all_positions.return_value = []
    bridge._client.get_orders.return_value = []
    return bridge


class TestPhase1UniverseInvariants:
    """Phase 1 universe membership and configuration invariants."""

    def test_t4_cvx_sector_attribution(self):
        """T4: CVX sector mapping exists and is attributed to ENERGY."""
        assert "CVX" in config.SECTOR_MAPPINGS, "CVX must be present in config.SECTOR_MAPPINGS"
        mapping = config.SECTOR_MAPPINGS["CVX"]
        assert mapping["sector"] == "ENERGY", f"Expected ENERGY, got {mapping.get('sector')}"
        assert mapping["source"] == "STATIC_CONFIG", f"Expected STATIC_CONFIG, got {mapping.get('source')}"

    def test_active_watchlist_invariants(self):
        """Phase 1: Active WATCHLIST must remain 20 symbols. CAT and LLY must NOT be promoted."""
        assert len(config.WATCHLIST) == 20, f"WATCHLIST count must remain 20 in Phase 1, got {len(config.WATCHLIST)}"
        assert "CAT" not in config.WATCHLIST, "CAT must not be in active WATCHLIST in Phase 1"
        assert "LLY" not in config.WATCHLIST, "LLY must not be in active WATCHLIST in Phase 1"
        assert "CVX" not in config.WATCHLIST, "CVX must not be in active WATCHLIST"

    def test_shadow_monitor_invariants(self):
        """Phase 1: SHADOW_MONITOR must contain CAT, LLY, and CVX."""
        shadow = set(config.SHADOW_MONITOR)
        assert "CVX" in shadow, "CVX must be in config.SHADOW_MONITOR"
        assert "CAT" in shadow, "CAT must remain in config.SHADOW_MONITOR in Phase 1"
        assert "LLY" in shadow, "LLY must remain in config.SHADOW_MONITOR in Phase 1"

    def test_daytrade_and_risk_policy_invariants(self):
        """Phase 1: Daytrade universe (17) and risk policies must remain completely frozen."""
        assert len(config.DAYTRADE_WATCHLIST) == 17, f"DAYTRADE_WATCHLIST must be 17, got {len(config.DAYTRADE_WATCHLIST)}"
        assert config.USE_DYNAMIC_SCANNER is False, "USE_DYNAMIC_SCANNER must be False"
        assert config.SECTOR_EXPOSURE_CAP_PCT == 0.25, "SECTOR_EXPOSURE_CAP_PCT must remain 0.25"
        assert config.MAX_CLUSTER_RISK_PCT == 0.0125, "MAX_CLUSTER_RISK_PCT must remain 0.0125"


class TestCVXZeroMutationShadowContract:
    """T3: Verify CVX signals are observable/loggable but trigger STRICTLY ZERO broker mutations."""

    def test_t3_a_cvx_buy_zero_mutation(self, mock_bridge):
        """T3-A: CVX BUY signal produces zero rotation side-effects and zero order submissions."""
        sig = Signal(
            symbol="CVX",
            action="BUY",
            confidence=0.85,
            rf_score=0.80,
            lstm_score=0.85,
            xgb_score=0.80,
            trans_score=0.80,
            tech_score=0.80,
            price=200.0,
            reason="[SHADOW MONITOR] strong test momentum (monitor-only, no order)",
            shadow_mode=True,
        )

        # 1. Rotation eligibility test: mock an open position that could be rotated out
        existing_pos = MagicMock(
            symbol="AAPL",
            side="long",
            qty=10,
            avg_entry_price=150.0,
            current_price=140.0,
            unrealized_pl=-100.0,
        )
        mock_bridge._client.get_all_positions.return_value = [existing_pos]

        # Execute rotation evaluation
        mock_bridge._evaluate_and_execute_rotation([sig])

        # Rotation must NOT trigger close_position or execute_signal
        mock_bridge._client.close_position.assert_not_called()
        mock_bridge._client.submit_order.assert_not_called()

        # 2. execute_all test: must skip CVX and submit 0 orders
        res = mock_bridge.execute_all([sig])
        assert res.get("placed", 0) == 0, f"Expected 0 placed orders, got {res.get('placed')}"
        assert res.get("skipped", 0) == 1, f"Expected 1 skipped signal, got {res.get('skipped')}"

        mock_bridge._client.submit_order.assert_not_called()
        mock_bridge._client.cancel_order.assert_not_called()
        mock_bridge._client.close_position.assert_not_called()

        # 3. Direct execute_signal test: hard-blocked at Layer 3
        executed = mock_bridge.execute_signal(sig)
        assert executed is False
        mock_bridge._client.submit_order.assert_not_called()

        # 4. Direct _place_open_bracket test: hard-blocked at boundary
        bracket_res = mock_bridge._place_open_bracket(
            symbol="CVX", action="BUY", qty=10, price=200.0
        )
        assert bracket_res.is_filled is False
        assert bracket_res.outcome == "REJECTED"
        mock_bridge._client.submit_order.assert_not_called()

    def test_t3_b_cvx_sell_zero_mutation(self, mock_bridge):
        """T3-B: CVX SELL signal must not cancel orders or mutate broker state."""
        sig = Signal(
            symbol="CVX",
            action="SELL",
            confidence=0.20,
            rf_score=0.20,
            lstm_score=0.20,
            xgb_score=0.20,
            trans_score=0.20,
            tech_score=0.20,
            price=200.0,
            reason="[SHADOW MONITOR] test sell signal (monitor-only, no order)",
            shadow_mode=True,
        )

        with patch.object(mock_bridge, "_cancel_working_entry_orders") as mock_cancel:
            res = mock_bridge.execute_all([sig])
            assert res.get("placed", 0) == 0
            assert res.get("skipped", 0) == 1
            mock_cancel.assert_not_called()

        mock_bridge._client.submit_order.assert_not_called()
        mock_bridge._client.cancel_order.assert_not_called()

        # Direct execute_signal test
        with patch.object(mock_bridge, "_cancel_working_entry_orders") as mock_cancel:
            executed = mock_bridge.execute_signal(sig)
            assert executed is False
            mock_cancel.assert_not_called()

    def test_t3_c_cvx_hold_zero_mutation(self, mock_bridge):
        """T3-C: CVX HOLD signal must not cancel orders or mutate broker state."""
        sig = Signal(
            symbol="CVX",
            action="HOLD",
            confidence=0.50,
            rf_score=0.50,
            lstm_score=0.50,
            xgb_score=0.50,
            trans_score=0.50,
            tech_score=0.50,
            price=200.0,
            reason="[SHADOW MONITOR] test hold signal (monitor-only, no order)",
            shadow_mode=True,
        )

        with patch.object(mock_bridge, "_cancel_working_entry_orders") as mock_cancel:
            res = mock_bridge.execute_all([sig])
            assert res.get("placed", 0) == 0
            assert res.get("skipped", 0) == 1
            mock_cancel.assert_not_called()

        mock_bridge._client.submit_order.assert_not_called()
        mock_bridge._client.cancel_order.assert_not_called()

        # Direct execute_signal test
        with patch.object(mock_bridge, "_cancel_working_entry_orders") as mock_cancel:
            executed = mock_bridge.execute_signal(sig)
            assert executed is False
            mock_cancel.assert_not_called()


class TestPhase2ModelCoverageQualification:
    """T5: Pre-promotion qualification test for Phase 2 target universe (22 symbols x 4 families)."""

    TARGET_PHASE2_UNIVERSE = [
        "SPY", "QQQ", "VTI",
        "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN",
        "GOOGL", "AVGO", "NFLX", "COST", "JPM", "BAC", "XOM", "UNH",
        "XLV", "V", "CAT", "LLY",
    ]

    def test_t5_complete_model_coverage_and_gates(self):
        """T5: Assert all 22 symbols in target qualification universe have 4/4 models + valid gates."""
        import joblib
        import torch
        import model_metrics

        # 1. Feature schema width invariant (Canonical 40-wide Swing feature schema)
        from ai_engine import StockRFEngine
        expected_width = len(StockRFEngine().feature_cols)
        assert expected_width == 40, f"Expected 40 canonical features, got {expected_width}"

        # 2. RF family coverage & input-width compatibility
        rf_path = config.ML_MODELS_FILE
        assert rf_path.exists(), "RF models file missing"
        rf_models = joblib.load(rf_path)
        rf_syms = set(rf_models.keys()) if isinstance(rf_models, dict) else set()
        for sym, model in rf_models.items():
            assert getattr(model, "n_features_in_", None) == expected_width, (
                f"RF model for {sym} feature width {getattr(model, 'n_features_in_', None)} != {expected_width}"
            )

        # 3. LSTM family coverage, input-size & scaler shape compatibility
        lstm_path = config.LSTM_CKPT_FILE
        assert lstm_path.exists(), "LSTM checkpoint missing"
        lstm_data = torch.load(lstm_path, map_location="cpu", weights_only=False)
        lstm_syms = set(lstm_data.get("symbols", [])) if isinstance(lstm_data, dict) else set()
        assert lstm_data.get("input_size") == expected_width, (
            f"LSTM checkpoint input_size {lstm_data.get('input_size')} != {expected_width}"
        )
        scalers_path = config.MODELS_DIR / "lstm_checkpoint.scalers.npz"
        if scalers_path.exists():
            import numpy as np
            npz = np.load(scalers_path)
            for sym in lstm_syms:
                if f"{sym}__mean" in npz.files:
                    assert npz[f"{sym}__mean"].shape == (expected_width,), (
                        f"LSTM scaler shape mismatch for {sym}: {npz[f'{sym}__mean'].shape} != ({expected_width},)"
                    )

        # 4. XGB family coverage & input-width compatibility
        xgb_path = config.MODELS_DIR / "xgb_models.joblib"
        assert xgb_path.exists(), "XGB models file missing"
        xgb_models = joblib.load(xgb_path)
        xgb_syms = set(xgb_models.keys()) if isinstance(xgb_models, dict) else set()
        for sym, model in xgb_models.items():
            assert getattr(model, "n_features_in_", None) == expected_width, (
                f"XGB model for {sym} feature width {getattr(model, 'n_features_in_', None)} != {expected_width}"
            )

        # 5. Transformer family coverage, input-size & scaler shape compatibility
        trans_path = config.MODELS_DIR / "transformer_checkpoint.pt"
        assert trans_path.exists(), "Transformer checkpoint missing"
        trans_data = torch.load(trans_path, map_location="cpu", weights_only=False)
        trans_syms = set(trans_data.get("symbols", [])) if isinstance(trans_data, dict) else set()
        assert trans_data.get("input_size") == expected_width, (
            f"Transformer checkpoint input_size {trans_data.get('input_size')} != {expected_width}"
        )
        trans_scalers_path = config.MODELS_DIR / "transformer_checkpoint.scalers.npz"
        if trans_scalers_path.exists():
            import numpy as np
            npz = np.load(trans_scalers_path)
            for sym in trans_syms:
                if f"{sym}__mean" in npz.files:
                    assert npz[f"{sym}__mean"].shape == (expected_width,), (
                        f"Transformer scaler shape mismatch for {sym}: {npz[f'{sym}__mean'].shape} != ({expected_width},)"
                    )

        # Verify 4/4 coverage for every symbol in target 22 universe
        missing_rf = set(self.TARGET_PHASE2_UNIVERSE) - rf_syms
        missing_lstm = set(self.TARGET_PHASE2_UNIVERSE) - lstm_syms
        missing_xgb = set(self.TARGET_PHASE2_UNIVERSE) - xgb_syms
        missing_trans = set(self.TARGET_PHASE2_UNIVERSE) - trans_syms

        assert not missing_rf, f"RF family missing models for: {missing_rf}"
        assert not missing_lstm, f"LSTM family missing models for: {missing_lstm}"
        assert not missing_xgb, f"XGB family missing models for: {missing_xgb}"
        assert not missing_trans, f"Transformer family missing models for: {missing_trans}"

        # 6. Canonical model gate verification
        gate_failures = {}
        for sym in self.TARGET_PHASE2_UNIVERSE:
            gate = model_metrics.evaluate_gate(sym)
            if not gate.ok:
                gate_failures[sym] = {"status": gate.status, "reason": gate.reason}

        assert not gate_failures, f"Canonical model gate failed for symbols: {gate_failures}"
