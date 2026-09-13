"""
tests/test_config_validation.py -- Tests for Alpaca runtime configuration validation.
"""
import pytest
import config
from config import (
    validate_alpaca_runtime_config,
    _require_positive_int,
    _require_non_negative_int,
    _require_positive_float,
)


class TestConfigValidation:
    """Test suite verifying strict config validation and bool-as-int rejection."""

    @pytest.mark.parametrize("invalid_value", [True, False, 0, -1, 1.5, "5", None])
    def test_chunk_size_rejects_invalid_values(self, invalid_value):
        with pytest.raises(ValueError, match="must be a positive integer"):
            _require_positive_int("ALPACA_CHUNK_SIZE", invalid_value)

    @pytest.mark.parametrize("invalid_value", [True, False, -1, 1.5, "5", None])
    def test_chunk_delay_rejects_invalid_values(self, invalid_value):
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            _require_non_negative_int("ALPACA_CHUNK_DELAY_MS", invalid_value)

    def test_chunk_delay_accepts_zero(self):
        assert _require_non_negative_int("ALPACA_CHUNK_DELAY_MS", 0) == 0

    @pytest.mark.parametrize("invalid_value", [True, False, 0, -1.0, "2.0", None])
    def test_chunk_timeout_rejects_invalid_values(self, invalid_value):
        with pytest.raises(ValueError, match="must be a positive float"):
            _require_positive_float("ALPACA_CHUNK_TIMEOUT_SECONDS", invalid_value)

    def test_subscription_tier_rejects_invalid(self, monkeypatch):
        monkeypatch.setattr(config, "ALPACA_SUBSCRIPTION_TIER", "pro_enterprise")
        with pytest.raises(ValueError, match="must be 'free' or 'unlimited'"):
            validate_alpaca_runtime_config()

    def test_valid_default_config_passes(self):
        # Must execute cleanly with default settings
        validate_alpaca_runtime_config()
