"""
backtest.py — Walk-forward backtest with broker-like position-state simulation.

Production hardening:
- Uses no future labels in evaluation rows; unknown future labels are excluded.
- Simulates daily position state instead of treating every signal as an isolated
  5-day trade. BUY/SELL behavior mirrors ibkr_bridge.py via the shared
  predictor.apply_position_rule_with_hold helper: long-only by default, SELL
  closes longs and does not open shorts unless config.ALLOW_SHORT=True. This is
  the same long-only / duplicate-entry logic ibkr_bridge.py enforces live.
- Horizon alignment: positions are held for at least config.MIN_HOLD_BARS bars,
  matching the ML_HORIZON forward-direction label the models are trained on.
- Uses the same threshold logic and technical-score helper as predictor.py.
- Backtest uses walk-forward RF + technical by default. Optional LSTM fold training
  is available with include_lstm=True / CLI --include-lstm, but it is slower.
- Includes transaction cost and slippage assumptions from config.
- Persists metrics JSON + per-row trade ledger CSV to REPORTS_DIR.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from ai_engine import build_rf, _subsample_train_indices
from data_manager import fetch_ohlcv, build_features, make_labels, get_feature_columns
from predictor import (
    action_from_confidence,
    technical_score_from_feature_row,
    weighted_blend,
    apply_position_rule_with_hold,
)

logger = logging.getLogger(__name__)
FEATURE_COLS = get_feature_columns()


def check_signal_parity() -> dict:
    """Confirm the backtest builds signals from the SAME source of truth as live.

    Phase 1 (explicit backtest/live parity). Pure, offline, deterministic: it
    verifies the blend / threshold / position helpers used here are the IDENTICAL
    objects exported by predictor (not a divergent copy), and that a numeric blend
    on sample scores matches between the two call sites. Returns a dict so the
    result is both machine-checkable (live-readiness gate, tests) and human-
    auditable (written into backtest_metrics.json as `signal_parity`).

    parity_ok says the *construction* is identical. Full *edge* parity also needs
    `backtest --include-lstm`, because LSTM is off by default in the backtest
    while the live ensemble uses it -- surfaced here as include_lstm_default.
    """
    import predictor as _p

    shared_blend = weighted_blend is _p.weighted_blend
    shared_action = action_from_confidence is _p.action_from_confidence
    shared_rule = apply_position_rule_with_hold is _p.apply_position_rule_with_hold

    numeric_ok = True
    for rf, lstm, tech in [(0.90, 0.80, 0.60), (0.30, 0.40, 0.50), (0.70, 0.20, 0.55)]:
        live = _p.weighted_blend(rf, True, lstm, True, tech)
        bt = weighted_blend(rf, True, lstm, True, tech)
        if abs(live - bt) > 1e-12:
            numeric_ok = False

    parity_ok = bool(shared_blend and shared_action and shared_rule and numeric_ok)
    return {
        "parity_ok": parity_ok,
        "shared_weighted_blend": bool(shared_blend),
        "shared_action_fn": bool(shared_action),
        "shared_position_rule": bool(shared_rule),
        "numeric_blend_match": bool(numeric_ok),
        "weights": {
            "rf": float(config.WEIGHT_RF),
            "lstm": float(config.WEIGHT_LSTM),
            "tech": float(config.WEIGHT_TECHNICAL),
        },
        "buy_threshold": float(config.BUY_THRESHOLD),
        "sell_threshold": float(config.SELL_THRESHOLD),
        "min_ml_models_for_signal": int(config.MIN_ML_MODELS_FOR_SIGNAL),
        "include_lstm_default": bool(config.BACKTEST_INCLUDE_LSTM),
        "note": (
            "Backtest and live share predictor.weighted_blend / action_from_confidence / "
            "apply_position_rule_with_hold. For full edge parity also run "
            "`backtest --include-lstm` (LSTM is off by default in the backtest)."
        ),
    }


def _sharpe(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    returns = np.asarray(daily_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / 252
    std = excess.std(ddof=1)
    if std <= 0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak == 0, 1, peak)
    return float(dd.min())


def _next_day_returns(feat: pd.DataFrame) -> pd.Series:
    """Return from close[t] to close[t+1], aligned to decision date t."""
    return feat["Close"].shift(-1) / feat["Close"] - 1.0


def _rf_scores_for_fold(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_te), 0.5, dtype=float)
    clf = build_rf()
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    classes = list(clf.classes_)
    if 1 not in classes:
        return np.full(len(X_te), 0.5, dtype=float)
    return proba[:, classes.index(1)].astype(float)


def _optional_lstm_scores_for_fold(
    X_all: np.ndarray,
    y_all: np.ndarray,
    start: int,
    step: int,
    purge_gap: int = 5,
) -> Tuple[np.ndarray, bool]:
    """
    Train a small fold-local LSTM without future leakage and score the test rows.
    This is intentionally optional because doing it for every fold/symbol is slow.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from lstm_engine import LSTMModel, _make_sequences
    except Exception:
        logger.exception("LSTM backtest dependencies unavailable")
        return np.full(step, 0.5, dtype=float), False

    window = int(config.LSTM_WINDOW)
    train_end = max(0, start - purge_gap)
    if train_end < window + 10 or len(np.unique(y_all[:train_end])) < 2:
        return np.full(step, 0.5, dtype=float), False

    try:
        X_train = X_all[:train_end].astype(np.float32)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        X_norm = ((X_all.astype(np.float32) - mean) / std).astype(np.float32)

        X_seq, y_seq = _make_sequences(X_norm[:train_end], y_all[:train_end].astype(np.float32), window)
        if len(X_seq) == 0 or len(np.unique(y_seq)) < 2:
            return np.full(step, 0.5, dtype=float), False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMModel(
            input_size=len(FEATURE_COLS),
            hidden_size=int(config.LSTM_HIDDEN),
            num_layers=int(config.LSTM_LAYERS),
            dropout=float(config.LSTM_DROPOUT),
            bidirectional=bool(getattr(config, "LSTM_BIDIRECTIONAL", False)),
        ).to(device)
        neg = float((y_seq == 0).sum())
        pos = float((y_seq == 1).sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LSTM_LR)

        dl = DataLoader(
            TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq.astype(np.float32))),
            batch_size=int(config.BACKTEST_LSTM_BATCH),
            shuffle=False,
        )
        epochs = int(config.BACKTEST_LSTM_EPOCHS)
        use_cuda_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

        for _epoch in range(max(1, epochs)):
            model.train()
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_cuda_amp):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        scores: List[float] = []
        model.eval()
        with torch.no_grad():
            for end in range(start, min(start + step, len(X_norm))):
                if end < window - 1:
                    scores.append(0.5)
                    continue
                seq = X_norm[end - window + 1:end + 1]
                t = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
                score = torch.sigmoid(model(t)).item()
                scores.append(float(score))

        if len(scores) < step:
            scores.extend([0.5] * (step - len(scores)))
        return np.asarray(scores[:step], dtype=float), True
    except Exception:
        logger.exception("Fold-local LSTM backtest failed")
        return np.full(step, 0.5, dtype=float), False


def run_backtest(
    symbols: Optional[List[str]] = None,
    train_min: int = 252,
    step: int = 21,
    verbose: bool = True,
    include_lstm: Optional[bool] = None,
    settings=None,
) -> dict:
    cfg = settings or config.get_settings()
    symbols = symbols or cfg.WATCHLIST
    include_lstm = bool(cfg.BACKTEST_INCLUDE_LSTM if include_lstm is None else include_lstm)
    allow_short = bool(cfg.ALLOW_SHORT)
    min_hold = int(cfg.MIN_HOLD_BARS)
    horizon = int(cfg.ML_HORIZON)
    purge_gap = int(getattr(cfg, "BACKTEST_PURGE_GAP", horizon))
    cost_per_order = float(cfg.BACKTEST_TRANSACTION_COST_PCT) + float(getattr(cfg, 'BACKTEST_SLIPPAGE_FRAC', cfg.BACKTEST_SLIPPAGE_PCT))

    all_rows: list = []
    symbol_metrics: dict = {}

    for symbol in symbols:
        try:
            df = fetch_ohlcv(symbol)
            feat = build_features(df)
            labels = make_labels(feat)
            next_ret = _next_day_returns(feat)

            # Need label for training target and next-day return for simulation.
            valid_idx = labels.dropna().index.intersection(next_ret.dropna().index)
            if len(valid_idx) == 0:
                logger.warning("No valid backtest rows for %s", symbol)
                continue

            X_all = feat.loc[valid_idx, FEATURE_COLS].values.astype(np.float32)
            y_all = labels.loc[valid_idx].values.astype(int)
            next_ret_all = next_ret.loc[valid_idx].values.astype(float)
            open_all = feat.loc[valid_idx, "Open"].values.astype(float)
            high_all = feat.loc[valid_idx, "High"].values.astype(float)
            low_all = feat.loc[valid_idx, "Low"].values.astype(float)
            close_all = feat.loc[valid_idx, "Close"].values.astype(float)
            idx_all = list(valid_idx)
            tech_all = np.asarray(
                [technical_score_from_feature_row(feat.loc[i]) for i in valid_idx],
                dtype=float,
            )

            if len(X_all) < train_min + step:
                logger.warning("Not enough data for backtest: %s (n=%d)", symbol, len(X_all))
                continue

            position = 0
            bars_held = 0
            entry_price = None
            equity = [1.0]
            daily_pnls: List[float] = []
            rows: List[dict] = []
            n_orders = 0
            n_hard_stop_exits = 0
            n_lstm_folds = 0
            hard_stop_pct = float(cfg.HARD_STOP_LOSS_PCT)
            prev_close = None

            train_window = int(getattr(cfg, "BACKTEST_TRAIN_WINDOW", 0))
            for start in range(train_min, len(X_all) - step + 1, step):
                train_end = max(0, start - purge_gap)
                if train_window > 0:
                    tr_start = max(0, train_end - train_window)
                    X_tr = X_all[tr_start:train_end]
                    y_tr = y_all[tr_start:train_end]
                else:
                    X_tr = X_all[:train_end]
                    y_tr = y_all[:train_end]
                X_te = X_all[start:start + step]
                ret_te = next_ret_all[start:start + step]
                idx_te = idx_all[start:start + step]
                tech_te = tech_all[start:start + step]
                open_te = open_all[start:start + step]
                high_te = high_all[start:start + step]
                low_te = low_all[start:start + step]
                close_te = close_all[start:start + step]

                if len(X_tr) < 10 or len(np.unique(y_tr)) < 2:
                    logger.debug("Only one class in train window: %s start=%d (train_end=%d)", symbol, start, train_end)
                    continue

                # Decorrelate overlapping forward-horizon labels by striding the
                # TRAINING rows only (ML_HORIZON), matching ai_engine's CV folds.
                # Test rows (X_te) are kept intact. Fall back to the full window
                # if the stride collapses the labels to a single class.
                sub = _subsample_train_indices(len(X_tr), horizon)
                X_tr_sub, y_tr_sub = X_tr[sub], y_tr[sub]
                if len(np.unique(y_tr_sub)) < 2:
                    X_tr_sub, y_tr_sub = X_tr, y_tr

                rf_scores = _rf_scores_for_fold(X_tr_sub, y_tr_sub, X_te)
                rf_ok = True
                if include_lstm:
                    lstm_scores, lstm_ok = _optional_lstm_scores_for_fold(
                        X_all, y_all, start, len(X_te), purge_gap=purge_gap
                    )
                    n_lstm_folds += int(lstm_ok)
                else:
                    lstm_scores = np.full(len(X_te), 0.5, dtype=float)
                    lstm_ok = False

                for dt, rf_score, lstm_score, tech_score, market_ret, open_px, high_px, low_px, close_px in zip(
                    idx_te, rf_scores, lstm_scores, tech_te, ret_te, open_te, high_te, low_te, close_te
                ):
                    open_px = float(open_px)
                    high_px = float(high_px)
                    low_px = float(low_px)
                    close_px = float(close_px)
                    if prev_close is None:
                        prev_close = float(close_all[start - 1]) if start > 0 else open_px

                    conf = weighted_blend(
                        float(rf_score), rf_ok,
                        float(lstm_score), lstm_ok,
                        float(tech_score),
                    )
                    signal = action_from_confidence(conf)
                    old_position = position
                    row_entry_price = entry_price
                    stop_px = (
                        row_entry_price * (1.0 - abs(hard_stop_pct))
                        if (row_entry_price is not None and hard_stop_pct and position > 0)
                        else None
                    )

                    # ── Phase 1: Open-time Gap Check ──────────────────
                    if position > 0 and stop_px is not None and open_px <= stop_px:
                        executed = True
                        n_orders += 1
                        n_hard_stop_exits += 1
                        gap_ret = (open_px / float(row_entry_price)) - 1.0
                        exec_note = f"hard-stop-gap ({gap_ret:.4f})"
                        exit_reason = exec_note
                        exit_price = round(open_px, 6)
                        gross_pnl = (open_px - prev_close) / prev_close
                        cost = cost_per_order
                        net_pnl = gross_pnl - cost
                        position = 0
                        entry_price = None
                        bars_held = 0

                    # ── Phase 2: Intrabar Stop Check ──────────────────
                    elif position > 0 and stop_px is not None and low_px <= stop_px:
                        executed = True
                        n_orders += 1
                        n_hard_stop_exits += 1
                        stop_ret = (stop_px / float(row_entry_price)) - 1.0
                        exec_note = f"hard-stop ({stop_ret:.4f})"
                        exit_reason = exec_note
                        exit_price = round(stop_px, 6)
                        gross_pnl = (stop_px - prev_close) / prev_close
                        cost = cost_per_order
                        net_pnl = gross_pnl - cost
                        position = 0
                        entry_price = None
                        bars_held = 0

                    # ── Phase 3: Close-time Signal Evaluation ─────────
                    else:
                        gross_pnl = float(position) * ((close_px - prev_close) / prev_close)

                        if position > 0:
                            bars_held += 1
                            if signal == "SELL":
                                if bars_held >= min_hold:
                                    executed = True
                                    n_orders += 1
                                    exec_note = "close-long"
                                    exit_reason = "close-long"
                                    exit_price = round(close_px, 6)
                                    cost = cost_per_order
                                    net_pnl = gross_pnl - cost
                                    position = 0
                                    entry_price = None
                                    bars_held = 0
                                else:
                                    executed = False
                                    exec_note = f"hold-long (min_hold {bars_held}/{min_hold})"
                                    exit_reason = ""
                                    exit_price = ""
                                    cost = 0.0
                                    net_pnl = gross_pnl
                            elif signal == "BUY":
                                executed = False
                                exec_note = "hold-long (already long)"
                                exit_reason = ""
                                exit_price = ""
                                cost = 0.0
                                net_pnl = gross_pnl
                            else:
                                executed = False
                                exec_note = "hold"
                                exit_reason = ""
                                exit_price = ""
                                cost = 0.0
                                net_pnl = gross_pnl

                        elif position < 0:
                            bars_held += 1
                            if signal == "BUY":
                                if bars_held >= min_hold:
                                    executed = True
                                    n_orders += 1
                                    exec_note = "cover-short"
                                    exit_reason = "cover-short"
                                    exit_price = round(close_px, 6)
                                    cost = cost_per_order
                                    net_pnl = gross_pnl - cost
                                    position = 0
                                    entry_price = None
                                    bars_held = 0
                                else:
                                    executed = False
                                    exec_note = f"hold-short (min_hold {bars_held}/{min_hold})"
                                    exit_reason = ""
                                    exit_price = ""
                                    cost = 0.0
                                    net_pnl = gross_pnl
                            elif signal == "SELL":
                                executed = False
                                exec_note = "hold-short (already short)"
                                exit_reason = ""
                                exit_price = ""
                                cost = 0.0
                                net_pnl = gross_pnl
                            else:
                                executed = False
                                exec_note = "hold"
                                exit_reason = ""
                                exit_price = ""
                                cost = 0.0
                                net_pnl = gross_pnl

                        else:  # position == 0 (flat)
                            if signal == "BUY":
                                executed = True
                                n_orders += 1
                                exec_note = "open-long"
                                exit_reason = ""
                                exit_price = ""
                                position = 1
                                entry_price = close_px
                                row_entry_price = close_px
                                bars_held = 1
                                cost = cost_per_order
                                net_pnl = 0.0 - cost
                                stop_px = (
                                    entry_price * (1.0 - abs(hard_stop_pct))
                                    if hard_stop_pct
                                    else None
                                )
                            elif signal == "SELL" and allow_short:
                                executed = True
                                n_orders += 1
                                exec_note = "open-short"
                                exit_reason = ""
                                exit_price = ""
                                position = -1
                                entry_price = close_px
                                row_entry_price = close_px
                                bars_held = 1
                                cost = cost_per_order
                                net_pnl = 0.0 - cost
                            else:
                                executed = False
                                exec_note = "skip-sell (flat, long-only)" if signal == "SELL" else "flat"
                                exit_reason = ""
                                exit_price = ""
                                cost = 0.0
                                net_pnl = 0.0

                    equity.append(equity[-1] * (1.0 + net_pnl))
                    daily_pnls.append(net_pnl)
                    prev_close = close_px

                    initial_stop_price = (
                        round(float(stop_px), 6)
                        if stop_px is not None
                        else ""
                    )
                    risk_per_share = (
                        round(float(row_entry_price) - float(initial_stop_price), 6)
                        if (row_entry_price is not None and initial_stop_price != "")
                        else ""
                    )
                    risk_source = "hard_stop" if (row_entry_price is not None and hard_stop_pct) else ""

                    rows.append({
                        "symbol": symbol,
                        "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                        "fold_start": int(start),
                        "signal": signal,
                        "confidence": round(float(conf), 6),
                        "rf_score": round(float(rf_score), 6),
                        "lstm_score": round(float(lstm_score), 6),
                        "lstm_included": bool(include_lstm and lstm_ok),
                        "tech_score": round(float(tech_score), 6),
                        "old_position": int(old_position),
                        "new_position": int(position),
                        "bars_held": int(bars_held),
                        "order_executed": bool(executed),
                        "execution_note": exec_note,
                        "exit_reason": exit_reason,
                        "hard_stop_exit": bool(exec_note.startswith("hard-stop")),
                        "entry_price": round(float(row_entry_price), 6) if row_entry_price is not None else "",
                        "initial_stop_price": initial_stop_price,
                        "risk_per_share": risk_per_share,
                        "risk_source": risk_source,
                        "close_price": round(float(close_px), 6),
                        "low_price": round(float(low_px), 6),
                        "exit_price": exit_price,
                        "next_day_return": round(float(market_ret), 8),
                        "gross_pnl": round(float(gross_pnl), 8),
                        "cost": round(float(cost), 8),
                        "net_pnl": round(float(net_pnl), 8),
                        "equity": round(float(equity[-1]), 8),
                    })

            if not rows:
                continue

            all_rows.extend(rows)
            eq_arr = np.asarray(equity, dtype=float)
            pnl_arr = np.asarray(daily_pnls, dtype=float)
            active_pnls = pnl_arr[np.asarray([r["new_position"] != 0 for r in rows], dtype=bool)]
            order_pnls = pnl_arr[np.asarray([r["order_executed"] for r in rows], dtype=bool)]

            total_return = float(eq_arr[-1] - 1.0)
            sharpe = _sharpe(pnl_arr)
            max_dd = _max_drawdown(eq_arr)
            win_rate = float((active_pnls > 0).mean()) if len(active_pnls) else 0.0
            order_win_rate = float((order_pnls > 0).mean()) if len(order_pnls) else 0.0

            symbol_metrics[symbol] = {
                "total_return": round(total_return, 4),
                "sharpe_ratio": round(sharpe, 4),
                "max_drawdown": round(max_dd, 4),
                "win_rate_active_days": round(win_rate, 4),
                "win_rate_order_days": round(order_win_rate, 4),
                "n_orders": int(n_orders),
                "n_hard_stop_exits": int(n_hard_stop_exits),
                "n_bars": int(len(rows)),
                "n_active_days": int(sum(1 for r in rows if r["new_position"] != 0)),
                "final_position": int(position),
                "lstm_folds_used": int(n_lstm_folds),
                "min_hold_bars": int(min_hold),
                "hard_stop_pct": round(hard_stop_pct, 6),
            }
            if verbose:
                logger.info(
                    "%-6s ret=%.2f%% sharpe=%.2f dd=%.2f%% orders=%d active_days=%d",
                    symbol, total_return * 100, sharpe, max_dd * 100,
                    n_orders, symbol_metrics[symbol] ["n_active_days"],
                )

        except Exception:
            logger.exception("Backtest failed for %s", symbol)

    if symbol_metrics:
        aggregate = {
            "symbols_tested": len(symbol_metrics),
            "total_trades": int(sum(m["n_orders"] for m in symbol_metrics.values())),
            "total_hard_stop_exits": int(sum(m["n_hard_stop_exits"] for m in symbol_metrics.values())),
            "total_bars": int(sum(m["n_bars"] for m in symbol_metrics.values())),
            "avg_total_return": round(float(np.mean([m["total_return"] for m in symbol_metrics.values()])), 4),
            "avg_sharpe": round(float(np.mean([m["sharpe_ratio"] for m in symbol_metrics.values()])), 4),
            "avg_win_rate": round(float(np.mean([m["win_rate_active_days"] for m in symbol_metrics.values()])), 4),
            "avg_max_drawdown": round(float(np.mean([m["max_drawdown"] for m in symbol_metrics.values()])), 4),
            "per_symbol": symbol_metrics,
        }
    else:
        aggregate = {
            "symbols_tested": 0,
            "total_trades": 0,
            "total_hard_stop_exits": 0,
            "total_bars": 0,
            "avg_total_return": 0.0,
            "avg_sharpe": 0.0,
            "avg_win_rate": 0.0,
            "avg_max_drawdown": 0.0,
            "per_symbol": {},
        }

    aggregate.update({
        "backtest_model": "RF_LSTM_TECHNICAL_WALK_FORWARD" if include_lstm else "RF_TECHNICAL_WALK_FORWARD",
        "include_lstm": bool(include_lstm),
        "purge_gap": int(purge_gap),
        # Phase 1: record explicit backtest/live signal-construction parity.
        "signal_parity": check_signal_parity(),
        "allow_short": bool(allow_short),
        "min_hold_bars": int(min_hold),
        "hard_stop_pct": round(float(cfg.HARD_STOP_LOSS_PCT), 6),
        "cost_per_order": round(float(cost_per_order), 8),
        "position_state_simulation": True,
        "note": (
            "Positions held >= MIN_HOLD_BARS to match ML_HORIZON forward-direction "
            "label and the live IBKR hold guard. Daily close-to-close returns used "
            "for held positions with purge gap and gap-down realistic stop fills."
        ),
    })

    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.REPORTS_DIR / "backtest_metrics.json"
    trades_path = cfg.REPORTS_DIR / "backtest_trades.csv"
    metrics_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    if all_rows:
        pd.DataFrame(all_rows).to_csv(trades_path, index=False)
    else:
        trades_path.write_text("", encoding="utf-8")

    logger.info(
        "Backtest complete | model=%s sharpe=%.2f win=%.1f%% short=%s hold=%d | %s",
        aggregate["backtest_model"],
        aggregate["avg_sharpe"],
        aggregate["avg_win_rate"] * 100,
        allow_short,
        min_hold,
        metrics_path,
    )
    return aggregate


if __name__ == "__main__":
    import argparse
    from logging_setup import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--train-min", type=int, default=252)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--include-lstm", action="store_true", help="Slow full-ensemble backtest")
    args = parser.parse_args()
    results = run_backtest(train_min=args.train_min, step=args.step, include_lstm=args.include_lstm)
    print(json.dumps(results, indent=2))