"""Offline regression probes using code extracted from the supplied source.

No project imports, credentials, networking, broker orders or state-file deletion.
These are isolated path tests, NOT the project's end-to-end pytest battery.
Run: python test_source_guards.py --source .
"""
import argparse
import ast
import datetime as dt
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--source", default=".")
args, remaining = parser.parse_known_args()
SOURCE = Path(args.source).resolve()
if not SOURCE.is_dir():
    SOURCE = Path(__file__).parent / args.source

logger = logging.getLogger("offline-probes")
logger.addHandler(logging.NullHandler())
NOW = dt.datetime(2026, 10, 15, 14, tzinfo=dt.timezone.utc)
TODAY = "2026-10-15"
CFG = NS(MIN_HISTORY_DAYS=3, CACHE_TTL_SECONDS=300, PRICE_PERIOD="5y",
         PRICE_INTERVAL="1d", SWING_MAX_DAILY_LOSS_PCT=.01,
         SWING_MAX_DAILY_LOSS_DOLLAR_CAP=300, MAX_DAILY_LOSS_DAYTRADE=300,
         SWING_ACCOUNT_ID="PA3R71I7JFDZ", SHADOW_MONITOR=[],
         ALLOW_VIX_PROXY_FALLBACK=False)


def module(name, **attrs):
    m = ModuleType(name)
    m.__dict__.update(attrs)
    return m


STRATEGIES = {
    "strategies": module("strategies"),
    "strategies.session": module("strategies.session", now_eastern=lambda: NOW),
    "strategies.regime_classifier": module(
        "strategies.regime_classifier",
        MarketRegime=NS(DATA_INVALID=NS(value="DATA_INVALID")),
        classify_market_regime=lambda **kw: NS(
            allows_new_entries=True, regime=NS(value="TEST"), reason="offline test")
    ),
    "strategies.intraday_risk": module(
        "strategies.intraday_risk", TECH_CORRELATED_CLUSTER=set(),
        check_soxx_tech_long_gate=lambda *a, **k: (True, "test")),
    "strategies.intraday_data": module("strategies.intraday_data"),
    "strategies.swing_sizing": module(
        "strategies.swing_sizing", calculate_swing_position_size=None,
        calculate_swing_stop_price=None, validate_swing_order_safety=None,
        build_canonical_swing_order_id=None, PositionIntent=None,
        SwingSizingRejected=type("SwingSizingRejected", (Exception,), {})),
}


def extract(name, class_name=None, members=None, global_names=None):
    tree = ast.parse((SOURCE / name).read_text(encoding="utf-8-sig"))
    if class_name:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (members is None or n.name in members)]
        nodes = [ast.ClassDef(name=class_name, bases=[], keywords=[], body=body,
                              decorator_list=[])]
    else:
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in members]
    # Future annotations avoids evaluating project-specific types in signatures.
    code_tree = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ] + nodes, type_ignores=[])
    env = dict(math=math, logger=logger, config_module=CFG, config=CFG, pd=pd,
               np=np, _dt=dt, time=time, Path=Path, os=__import__("os"))
    env.update(global_names or {})
    exec(compile(ast.fix_missing_locations(code_tree), name, "exec"), env)
    return env


BRIDGE_METHODS = {
    "_positive_finite_equity", "_restore_session_baseline",
    "snapshot_start_of_day_equity", "_is_daily_loss_halted",
    "get_order_quote", "validate_quote", "execute_signal"
}


class Q:
    def __init__(self, ok, price, source, reason):
        self.ok, self.price, self.source, self.reason = ok, price, source, reason


BENV = extract("alpaca_bridge.py", "AlpacaBridge", BRIDGE_METHODS, dict(
    PersistedRiskStateInvalidError=type("PersistedRiskStateInvalidError", (Exception,), {}),
    QuoteDecision=Q, StockLatestQuoteRequest=lambda **kw: kw,
    StockLatestTradeRequest=lambda **kw: kw, normalize_enum_value=lambda x: x,
    check_regular_market_hours=lambda client: (True, "test"),
))
B = BENV["AlpacaBridge"]


def baseline_state(value=10000.0, date=TODAY, source="broker_last_equity"):
    return dict(account_id="PA3R71I7JFDZ", date=date, start_of_day_equity=value,
                suspended=False, provenance=dict(source=source))


def bridge(state=None):
    b = B()
    b.mode = "swing"
    b._account_id = "PA3R71I7JFDZ"
    b._start_of_day_equity = 10000.0
    b._start_of_day_equity_date = TODAY
    b._start_of_day_equity_source = "broker_last_equity"
    b._daytrade_suspended = False
    b._load_daytrade_risk_state = lambda acct: state or {}
    b._call_with_retry = lambda fn: fn()
    b.saved = []
    b._save_daytrade_risk_state = lambda *a, **k: b.saved.append((a, k))
    b._client = NS(get_account=lambda: NS(last_equity=10000, equity=9950))
    b.entry_gate = None
    b.get_net_liquidation = lambda: 9950.0
    return b


def run_connect_baseline(b, acct, state):
    """Execute the actual connect() baseline block, not a rewritten model."""
    text = (SOURCE / "alpaca_bridge.py").read_text(encoding="utf-8-sig")
    if '            self._restore_session_baseline(acct, state, et_date, self._account_id)' in text:
        return b._restore_session_baseline(acct, state, TODAY, b._account_id)
    import textwrap
    begin = text.index('            broker_last_eq = float(getattr(acct, "last_equity", 0.0) or 0.0)')
    end = text.index("            # Attach FlightRecorderHandler", begin)
    exec(compile(textwrap.dedent(text[begin:end]), "connect_baseline_block", "exec"),
         dict(self=b, acct=acct, state=state, et_date=TODAY, logger=logger))
    return b._start_of_day_equity


def fake_df():
    return pd.DataFrame(dict(Open=[10.]*5, High=[11.]*5, Low=[9.]*5,
                             Close=[10.]*5, Volume=[100.]*5),
                        index=pd.date_range("2026-10-01", periods=5))


def manager_env(yf_result=None, yf_error=False):
    def history(**kw):
        if yf_error:
            raise RuntimeError("offline source failure")
        return yf_result.copy()
    return extract("data_manager.py", members={"with_exponential_backoff", "fetch_ohlcv"}, global_names=dict(
        yf=NS(Ticker=lambda symbol: NS(history=history)), _PROVIDER=None, _cache={},
        functools=__import__("functools"), MAX_FETCH_RETRIES=3, RETRY_BASE_DELAY=0,
        RETRY_BACKOFF_MULT=1, RETRY_MAX_DELAY=0,
        time=NS(time=time.time, sleep=lambda _: None)))


class SourceGuards(unittest.TestCase):
    def setUp(self):
        self.mods = patch.dict(sys.modules, STRATEGIES)
        self.mods.start()

    def tearDown(self):
        self.mods.stop()

    def test_connect_does_not_stamp_current_equity_as_broker_last_equity(self):
        b = bridge()
        result = run_connect_baseline(b, NS(last_equity=0, equity=10000), {})
        self.assertEqual(result, 0)
        self.assertNotEqual(b._start_of_day_equity_source, "broker_last_equity")
        self.assertEqual(b.saved, [])

    def test_reconnect_preserves_existing_session_baseline(self):
        b = bridge()
        value = run_connect_baseline(b, NS(last_equity=9900, equity=9800), baseline_state())
        self.assertEqual(value, 10000)

    def test_connect_rejects_non_finite_broker_baselines(self):
        for bad in (float("nan"), float("inf"), -1, None):
            with self.subTest(value=bad):
                b = bridge()
                self.assertEqual(run_connect_baseline(b, NS(last_equity=bad, equity=10000), {}), 0)

    def test_snapshot_does_not_accept_infinite_memory_baseline(self):
        b = bridge()
        b._start_of_day_equity = float("inf")
        b._client = NS(get_account=lambda: NS(last_equity=0, equity=10000))
        self.assertEqual(b.snapshot_start_of_day_equity(), 0)

    def test_snapshot_does_not_accept_infinite_broker_baseline(self):
        b = bridge()
        b._start_of_day_equity = 0
        b._client = NS(get_account=lambda: NS(last_equity="inf", equity=10000))
        self.assertEqual(b.snapshot_start_of_day_equity(), 0)

    def test_daily_check_does_not_trust_unverified_positive_memory(self):
        b = bridge()
        b._start_of_day_equity_source = "unverified"
        b._client = NS(get_account=lambda: NS(last_equity=0, equity=9950))
        self.assertTrue(b._is_daily_loss_halted())

    def test_daily_check_does_not_trust_previous_date_memory(self):
        b = bridge()
        b._start_of_day_equity_date = "2026-10-14"
        b._client = NS(get_account=lambda: NS(last_equity=0, equity=9950))
        self.assertTrue(b._is_daily_loss_halted())

    def test_control_healthy_equity_allows_daily_gate(self):
        self.assertFalse(bridge(baseline_state())._is_daily_loss_halted())

    def test_control_loss_breach_blocks_daily_gate(self):
        b = bridge(baseline_state())
        b.get_net_liquidation = lambda: 9600
        self.assertTrue(b._is_daily_loss_halted())

    def test_control_invalid_current_equity_blocks(self):
        for value in (0, -1, float("nan"), float("inf")):
            b = bridge(baseline_state())
            b.get_net_liquidation = lambda: value
            self.assertTrue(b._is_daily_loss_halted())

    def execution_bridge(self):
        b = bridge(baseline_state())
        b._c = lambda k, default: default
        b.get_position = lambda s: 0
        b.get_order_quote = lambda s: Q(True, 100, "alpaca", "ok")
        b._cfg = CFG
        b._enforce_market_hours_test = False
        import threading
        b._swing_dispatch_lock = threading.Lock()
        acct = NS(account_number=b._account_id, equity=9950, last_equity=10000,
                  buying_power=9950)
        b._client = NS(get_account=lambda: acct, get_all_positions=lambda: [])
        b._get_active_orders = lambda: []
        b.audit_called = False
        def audit(*a):
            b.audit_called = True
            raise RuntimeError("Offline sentinel: never proceed to order submission")
        b.audit_swing_portfolio_safety = audit
        b.entry_gate = NS(
            check_daily_loss_halt_state=lambda *a: (False, ""),
            daily_loss_blocked=lambda **kw: (False, ""),
            entry_blocked=lambda *a, **kw: (False, ""))
        return b

    def signal(self):
        return NS(symbol="BAC", action="BUY", confidence=.9)

    def test_direct_buy_rejects_fallback_execution_price(self):
        b = self.execution_bridge()
        b.get_order_quote = lambda s: Q(True, 100, "alpaca_iex_trade_fallback", "fallback")
        self.assertFalse(b.execute_signal(self.signal()))
        self.assertFalse(b.audit_called)

    def test_direct_buy_does_not_replace_missing_sod_with_equity(self):
        b = self.execution_bridge()
        b._start_of_day_equity = 0
        b._load_daytrade_risk_state = lambda acct: {}
        b._client.get_account = lambda: NS(account_number=b._account_id,
                                         equity=9950, buying_power=9950)
        seen = []
        b.entry_gate.daily_loss_blocked = lambda **kw: (seen.append(kw) or False, "")
        self.assertFalse(b.execute_signal(self.signal()))
        self.assertEqual(seen, [], "Invalid SOD must stop before a manufactured P&L reaches the gate")
        self.assertFalse(b.audit_called)

    def test_direct_buy_blocks_unknown_position_inventory(self):
        b = self.execution_bridge()
        def fail():
            raise RuntimeError("broker positions API unavailable")
        b._client.get_all_positions = fail
        self.assertFalse(b.execute_signal(self.signal()))
        self.assertFalse(b.audit_called)

    def test_direct_buy_blocks_unknown_open_order_inventory(self):
        b = self.execution_bridge()
        def fail():
            raise RuntimeError("broker orders API unavailable")
        b._get_active_orders = fail
        self.assertFalse(b.execute_signal(self.signal()))
        self.assertFalse(b.audit_called)

    def quote_bridge(self, age=0, bid=99.9, ask=100.1, timestamp=True):
        b = bridge()
        b._require_connection = lambda: None
        b._c = lambda k, default: default
        stamp = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age)
        q = NS(bid_price=bid, ask_price=ask, timestamp=stamp if timestamp else None)
        b._data_client = NS(get_stock_latest_quote=lambda req: {"BAC": q},
                            get_stock_latest_trade=lambda req: {"BAC": NS(price=100)})
        return b

    def test_execution_quote_rejects_stale_top_of_book(self):
        self.assertFalse(self.quote_bridge(age=600).validate_quote("BAC")[0])

    def test_execution_quote_rejects_missing_timestamp(self):
        self.assertFalse(self.quote_bridge(timestamp=False).validate_quote("BAC")[0])

    def test_control_fresh_two_sided_quote_passes(self):
        self.assertTrue(self.quote_bridge(age=0).validate_quote("BAC")[0])

    def test_control_fallback_still_available_for_valuation(self):
        q = self.quote_bridge(age=600, bid=90, ask=110).get_order_quote("BAC")
        self.assertTrue(q.ok)
        self.assertEqual(q.source, "alpaca_iex_trade_fallback")

    def test_manager_drops_infinite_ohlc(self):
        df = fake_df()
        df.iloc[0, df.columns.get_loc("High")] = np.inf
        result = manager_env(df)["fetch_ohlcv"]("BAC")
        self.assertEqual(len(result), 4)
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_provider_drops_infinite_ohlc(self):
        df = fake_df()
        df.iloc[0, df.columns.get_loc("High")] = np.inf
        env = extract("data_providers.py", "MultiSourceDataProvider", {"_clean_and_validate"})
        result = env["MultiSourceDataProvider"]()._clean_and_validate("BAC", df)
        self.assertEqual(len(result), 4)

    def test_manager_preserves_actual_yfinance_source(self):
        result = manager_env(fake_df())["fetch_ohlcv"]("BAC")
        self.assertEqual(result.attrs["volume_source"], "yfinance_consolidated")

    def test_manager_never_substitutes_daily_data_for_five_minute_request(self):
        calls = []
        STRATEGIES["strategies.intraday_data"].fetch_daily_alpaca = lambda *a, **k: (
            calls.append(1) or fake_df())
        with self.assertRaises(ValueError):
            manager_env(yf_error=True)["fetch_ohlcv"]("BAC", interval="5m")
        self.assertEqual(calls, [])

    def test_provider_never_substitutes_daily_data_for_five_minute_request(self):
        calls = []
        STRATEGIES["strategies.intraday_data"].fetch_daily_alpaca = lambda *a, **k: (
            calls.append(1) or fake_df())
        with tempfile.TemporaryDirectory() as d:
            env = extract("data_providers.py", "MultiSourceDataProvider",
                          {"fetch", "_clean_and_validate"},
                          dict(CACHE_DIR=Path(d), DataProviderError=ValueError))
            p = env["MultiSourceDataProvider"]()
            p.ib_bridge = p.polygon_api_key = None
            def fail(*a):
                raise RuntimeError("offline source failure")
            p._fetch_yfinance = fail
            with self.assertRaises(ValueError):
                p.fetch("BAC", interval="5m", force_refresh=True)
            self.assertEqual(calls, [])

    def test_provider_cache_does_not_bypass_validation(self):
        with tempfile.TemporaryDirectory() as d:
            df = fake_df()
            df.iloc[0, df.columns.get_loc("High")] = np.inf
            df.to_csv(Path(d) / "BAC_5y_1d.csv")
            env = extract("data_providers.py", "MultiSourceDataProvider",
                          {"fetch", "_clean_and_validate"},
                          dict(CACHE_DIR=Path(d), DataProviderError=ValueError))
            result = env["MultiSourceDataProvider"]().fetch("BAC")
            self.assertEqual(len(result), 4)

    def test_spy_failure_does_not_skip_independent_vix_request(self):
        calls = []
        def fetch(symbol):
            calls.append(symbol)
            if symbol == "SPY":
                raise RuntimeError("SPY unavailable")
            return pd.DataFrame({"Close": [20.]})
        env = extract("predictor.py", members={"get_macro_benchmark_indicators"},
                      global_names=dict(fetch_ohlcv=fetch, compute_adx=lambda *a: 20))
        result = env["get_macro_benchmark_indicators"]()
        self.assertIn("^VIX", calls)
        self.assertEqual(result["vix"], 20)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SourceGuards)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(source=str(SOURCE), tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  failed_tests=[str(t) for t, _ in result.failures + result.errors])
    print(json.dumps(report, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
