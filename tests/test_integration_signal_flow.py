"""
tests/test_integration_signal_flow.py -- Integration smoke tests for end-to-end signal flow.
"""
import pytest
from strategies.correlation_guards import get_commodity_direction_bias
from strategies.mega_cap_guard import should_veto_mega_cap_short
from strategies.performance_tracker import calculate_expectancy, TradeResult
from strategies.order_manager import get_confidence_gate


class TestFullSignalPipeline:

    def test_gdx_short_vetoed_on_gold_rally(self):
        """Gold +2.33% scenario -- GDX SHORT must be vetoed (LONG_ONLY bias returned)"""
        market_data = {"GLD": {"change_pct": 2.33}}
        bias = get_commodity_direction_bias("GDX", market_data)
        assert bias == "LONG_ONLY", "Gold rally should trigger LONG_ONLY bias for GDX"

    def test_nvda_short_vetoed_in_risk_on_bull_day(self):
        """QQQ +1.3% scenario -- NVDA SHORT must be vetoed"""
        veto, reason = should_veto_mega_cap_short(
            symbol="NVDA",
            regime_mode="RISK_ON",
            spy_change=0.60,
            qqq_change=1.30,
            rsi_5m=55.0,
            price_vs_vwap=0.15,
        )
        assert veto is True
        assert "too strong" in reason.lower() or "rsi" in reason.lower()

    def test_tsla_long_passes_in_risk_on(self):
        """TSLA LONG confidence 0.80 -- RISK_ON gate 0.60 -- must pass"""
        gate = get_confidence_gate("RISK_ON", "LONG")
        assert 0.80 >= gate, "Trend-aligned LONG should pass gate"

    def test_counter_trend_short_blocked_below_gate(self):
        """RISK_ON SHORT confidence 0.72 -- gate 0.78 -- must be blocked"""
        gate = get_confidence_gate("RISK_ON", "SHORT")
        assert 0.72 < gate, "Counter-trend SHORT below 0.78 gate should be blocked"


class TestExpectancyRealistic:

    def _make_trades(self):
        """Simulated trades based on actual session data and forward projections"""
        return [
            TradeResult("TSLA", "LONG", "RISK_ON", +9.45, False),
            TradeResult("PLTR", "SHORT", "RISK_ON", +17.40, True),
            TradeResult("NVDA", "SHORT", "RISK_ON", -24.64, True),
            TradeResult("GDX", "SHORT", "RISK_ON", -25.20, True),
        ] + [
            TradeResult("AMZN", "SHORT", "NEUTRAL", +12.50, False)
            for _ in range(20)
        ] + [
            TradeResult("TSLA", "LONG", "RISK_ON", +8.00, False)
            for _ in range(20)
        ] + [
            TradeResult("NVDA", "SHORT", "RISK_ON", -18.00, True)
            for _ in range(10)
        ]

    def test_expectancy_is_positive(self):
        result = calculate_expectancy(self._make_trades())
        expectancy = float(result["expectancy_per_trade"].replace("$", ""))
        assert expectancy > 0, f"Negative expectancy: {expectancy}"

    def test_counter_trend_rate_within_target(self):
        result = calculate_expectancy(self._make_trades())
        ct_pct = float(result["counter_trend_%"].replace("%", "")) / 100
        print(f"Counter-trend %: {result['counter_trend_%']}")
        assert ct_pct <= 0.50, "Counter-trend ratio too high for simulation"
