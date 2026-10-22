"""Regression coverage for Swing sizing vs the $500 order minimum."""

from strategies.swing_sizing import (
    MIN_ORDER_NOTIONAL,
    calculate_swing_position_size,
)


def test_subminimum_bac_order_rejected_inside_sizer_before_validation():
    # Mirrors the observed shape: q_portfolio=1 under the 28% cap.
    res = calculate_swing_position_size(
        entry_price=50.565,
        stop_price=49.50,
        account_equity=100_000.0,
        available_buying_power=68_000.0,
        existing_portfolio_exposure=27_925.00,
        existing_symbol_exposure=0.0,
        adv_shares_20=1_000_000.0,
    )

    assert MIN_ORDER_NOTIONAL == 500.0
    assert res.rejected is True
    assert res.shares == 0
    assert res.binding_constraint == "MINIMUM_NOTIONAL"
    assert res.q_portfolio == 1
    assert "MIN_ORDER_NOTIONAL_UNSATISFIABLE" in res.rejection_reason
    assert "$50.56" in res.rejection_reason


def test_subminimum_rejection_does_not_override_risk_caps_or_fake_a_trade():
    # Even with ample buying power and loose other caps, a $50 one-share order
    # cannot be dispatched when the remaining portfolio budget is under $500.
    res = calculate_swing_position_size(
        entry_price=50.0,
        stop_price=49.0,
        account_equity=100_000.0,
        available_buying_power=50_000.0,
        existing_portfolio_exposure=27_550.0,  # $450 remaining under $28k cap
        adv_shares_20=1_000_000.0,
    )

    assert res.rejected is True
    assert res.shares == 0
    assert res.q_portfolio == 9
    assert res.gross_notional == 0.0


def test_sizer_still_allows_valid_order_at_or_above_minimum():
    res = calculate_swing_position_size(
        entry_price=50.0,
        stop_price=49.5,
        account_equity=100_000.0,
        available_buying_power=50_000.0,
        existing_portfolio_exposure=27_400.0,  # $600 budget -> 12 shares
        adv_shares_20=1_000_000.0,
    )

    assert res.rejected is False
    assert res.shares >= 10
    assert res.gross_notional >= MIN_ORDER_NOTIONAL


def test_empirical_oct19_deadlock_replay_high_price_vs_subminimum():
    """
    Replay of the 2026-10-19 20:31 ICT sizing evaluation:
    Account equity = $99,769.05, 28% Gross Cap = $27,935.33.
    Existing exposure from 4 live positions:
      - AMZN (39 sh @ $256.00) = $9,984.00
      - NVDA (45 sh @ $216.70) = $9,751.50
      - MSFT (8 sh @ $531.64)  = $4,253.12
      - JPM (11 sh @ $338.86)  = $3,727.46
      Total Market Exposure     = $27,716.08 (27.78% of equity)
    Remaining portfolio budget under 28% cap = $27,935.33 - $27,716.08 = $219.25.

    Proves:
    1. Higher-priced tickers (> $219.25) get q_portfolio = 0 and ZERO_SHARES.
    2. Lower-priced tickers (< $219.25) fit whole share capacity (BAC gets q_portfolio = 4,
       gross notional = 4 * $51.32 = $205.28) but are cleanly rejected by the $500 minimum notional constraint.
    3. Neither can be admitted without breaching the 28% cap or the $500 minimum.
    """
    equity = 99_769.05
    existing_port_exp = 27_716.08  # 27.780% of equity
    buying_power = 68_000.0

    # 1. AVGO @ $334.78
    res_avgo = calculate_swing_position_size(
        entry_price=334.78,
        stop_price=328.00,
        account_equity=equity,
        available_buying_power=buying_power,
        existing_portfolio_exposure=existing_port_exp,
        adv_shares_20=5_000_000.0,
    )
    assert res_avgo.rejected is True
    assert res_avgo.shares == 0
    assert res_avgo.q_portfolio == 0
    assert res_avgo.binding_constraint == "ZERO_SHARES"

    # 2. QQQ @ $719.14
    res_qqq = calculate_swing_position_size(
        entry_price=719.14,
        stop_price=710.00,
        account_equity=equity,
        available_buying_power=buying_power,
        existing_portfolio_exposure=existing_port_exp,
        adv_shares_20=30_000_000.0,
    )
    assert res_qqq.rejected is True
    assert res_qqq.shares == 0
    assert res_qqq.q_portfolio == 0
    assert res_qqq.binding_constraint == "ZERO_SHARES"

    # 3. GOOGL @ $344.08
    res_googl = calculate_swing_position_size(
        entry_price=344.08,
        stop_price=338.00,
        account_equity=equity,
        available_buying_power=buying_power,
        existing_portfolio_exposure=existing_port_exp,
        adv_shares_20=15_000_000.0,
    )
    assert res_googl.rejected is True
    assert res_googl.shares == 0
    assert res_googl.q_portfolio == 0
    assert res_googl.binding_constraint == "ZERO_SHARES"

    # 4. BAC @ $51.32
    res_bac = calculate_swing_position_size(
        entry_price=51.32,
        stop_price=50.30,
        account_equity=equity,
        available_buying_power=buying_power,
        existing_portfolio_exposure=existing_port_exp,
        adv_shares_20=25_000_000.0,
    )
    assert res_bac.rejected is True
    assert res_bac.shares == 0
    assert res_bac.q_portfolio == 4
    assert res_bac.binding_constraint == "MINIMUM_NOTIONAL"
    assert "MIN_ORDER_NOTIONAL_UNSATISFIABLE" in res_bac.rejection_reason

