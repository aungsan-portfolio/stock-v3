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

    def test_swing_risk_defaults_in_subprocess(self):
        import subprocess
        import sys
        code = (
            "import config\n"
            "assert config.SWING_MAX_DAILY_LOSS_PCT == 0.01\n"
            "assert config.SWING_MAX_DAILY_LOSS_DOLLAR_CAP == 300.00\n"
            "assert config.SWING_PROTECTED_SYMBOLS == {'SPY', 'QQQ'}\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, f"Subprocess failed: {res.stderr}"

    def test_swing_risk_env_overrides_in_subprocess(self):
        import os
        import subprocess
        import sys
        env = os.environ.copy()
        env["SWING_MAX_DAILY_LOSS_PCT"] = "0.03"
        env["SWING_MAX_DAILY_LOSS_DOLLAR_CAP"] = "500.00"
        env["SWING_PROTECTED_SYMBOLS"] = "AAPL, MSFT ,GOOG "
        code = (
            "import config\n"
            "assert abs(config.SWING_MAX_DAILY_LOSS_PCT - 0.03) < 1e-6\n"
            "assert abs(config.SWING_MAX_DAILY_LOSS_DOLLAR_CAP - 500.00) < 1e-6\n"
            "assert config.SWING_PROTECTED_SYMBOLS == {'AAPL', 'MSFT', 'GOOG'}\n"
        )
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert res.returncode == 0, f"Subprocess failed: {res.stderr}"

    def test_swing_risk_dollar_cap_zero_rejected_option_c(self):
        import os
        import subprocess
        import sys
        env = os.environ.copy()
        env["SWING_MAX_DAILY_LOSS_DOLLAR_CAP"] = "0.0"
        code = "import config"
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert res.returncode != 0
        assert "SWING_MAX_DAILY_LOSS_DOLLAR_CAP must be greater than 0" in res.stderr

    @pytest.mark.parametrize(
        "env_var,val,err_match",
        [
            ("SWING_MAX_DAILY_LOSS_PCT", "0.15", "SWING_MAX_DAILY_LOSS_PCT must be between 0 and 0.10"),
            ("SWING_MAX_DAILY_LOSS_PCT", "-0.01", "SWING_MAX_DAILY_LOSS_PCT must be between 0 and 0.10"),
            ("SWING_MAX_DAILY_LOSS_DOLLAR_CAP", "-1.0", "SWING_MAX_DAILY_LOSS_DOLLAR_CAP must be greater than 0"),
            ("SWING_MAX_DAILY_LOSS_DOLLAR_CAP", "0.0", "SWING_MAX_DAILY_LOSS_DOLLAR_CAP must be greater than 0"),
            ("SWING_MAX_DAILY_LOSS_DOLLAR_CAP", "-50.0", "SWING_MAX_DAILY_LOSS_DOLLAR_CAP must be greater than 0"),
        ],
    )
    def test_swing_risk_validation_errors_in_subprocess(self, env_var, val, err_match):
        import os
        import subprocess
        import sys
        env = os.environ.copy()
        env[env_var] = val
        code = "import config"
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert res.returncode != 0
        assert err_match in res.stderr

    def test_single_definition_of_swing_risk_configs(self):
        import ast
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "config.py"
        tree = ast.parse(config_path.read_text(encoding="utf-8"))

        target_names = {
            "SWING_MAX_DAILY_LOSS_PCT",
            "SWING_MAX_DAILY_LOSS_DOLLAR_CAP",
            "SWING_PROTECTED_SYMBOLS",
        }
        counts = {name: 0 for name in target_names}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in counts:
                        counts[target.id] += 1

        for name, count in counts.items():
            assert count == 1, f"Expected {name} to be assigned exactly once, found {count} assignments."

    def test_max_daily_loss_pct_default_and_override_in_subprocess(self):
        import subprocess
        import sys
        import os

        # 1. Default value is 0.9 percentage points
        code_default = "import config\nassert config.MAX_DAILY_LOSS_PCT == 0.9\n"
        res = subprocess.run([sys.executable, "-c", code_default], capture_output=True, text=True)
        assert res.returncode == 0, f"Default check failed: {res.stderr}"

        # 2. Env override to 1.0
        env = os.environ.copy()
        env["MAX_DAILY_LOSS_PCT"] = "1.0"
        code_override = "import config\nassert abs(config.MAX_DAILY_LOSS_PCT - 1.0) < 1e-6\n"
        res_env = subprocess.run([sys.executable, "-c", code_override], env=env, capture_output=True, text=True)
        assert res_env.returncode == 0, f"Env override check failed: {res_env.stderr}"

    @pytest.mark.parametrize("bad_val", ["0.0", "-1.0", "10.5", "100.0"])
    def test_max_daily_loss_pct_bounds_validation_in_subprocess(self, bad_val):
        import subprocess
        import sys
        import os
        env = os.environ.copy()
        env["MAX_DAILY_LOSS_PCT"] = bad_val
        code = "import config"
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert res.returncode != 0
        assert "MAX_DAILY_LOSS_PCT must be greater than 0 and no greater than 10.0 percentage points" in res.stderr

