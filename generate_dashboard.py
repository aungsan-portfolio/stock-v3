"""
generate_dashboard.py — Compiled static HTML dashboard generator.
Reads JSON reports and compiles a self-contained interactive HTML dashboard.
"""
import json
import logging
import os
import sys
from pathlib import Path

import config
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Engine Pro V3 Dashboard</title>
    <style>
        :root {
            color-scheme: light;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-sub: #64748b;
            --border: #e2e8f0;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }
        h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 700;
        }
        .timestamp {
            color: var(--text-sub);
            font-size: 14px;
        }
        .grid-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .card-title {
            color: var(--text-sub);
            font-size: 12px;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 28px;
            font-weight: 700;
            margin: 0;
        }
        .card-value.pos { color: var(--success); }
        .card-value.neg { color: var(--danger); }

        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border);
            margin-bottom: 20px;
            gap: 8px;
        }
        .tab-btn {
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            padding: 10px 16px;
            font-size: 15px;
            font-weight: 600;
            color: var(--text-sub);
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab-btn:hover {
            color: var(--text-main);
        }
        .tab-btn.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        th, td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background-color: #f1f5f9;
            font-weight: 600;
            color: var(--text-sub);
            font-size: 13px;
        }
        tr:last-child td {
            border-bottom: none;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .status-badge.healthy { background: #d1fae5; color: #065f46; }
        .status-badge.stale { background: #fef3c7; color: #92400e; }
        .status-badge.missing { background: #fee2e2; color: #991b1b; }

        .chart-container {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }
        .chart-header {
            margin-bottom: 16px;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Stock Engine Pro V3</h1>
                <div class="timestamp">Compiled Dashboard &bull; Live Updates</div>
            </div>
            <div class="timestamp">Generated on: <span id="gen-time">__GEN_TIME__</span></div>
        </header>

        <!-- Summary Cards -->
        <div class="grid-cards">
            <div class="card">
                <div class="card-title">Avg Backtest Return</div>
                <div class="card-value __AVG_RET_CLASS__">__AVG_RET__</div>
            </div>
            <div class="card">
                <div class="card-title">Avg Backtest Sharpe</div>
                <div class="card-value">__AVG_SHARPE__</div>
            </div>
            <div class="card">
                <div class="card-title">Max Backtest Drawdown</div>
                <div class="card-value neg">__AVG_DD__</div>
            </div>
            <div class="card">
                <div class="card-title">Paper Live P&L</div>
                <div class="card-value __LIVE_PNL_CLASS__">__LIVE_PNL__</div>
            </div>
            <div class="card">
                <div class="card-title">Paper Open Positions</div>
                <div class="card-value">__OPEN_POS_COUNT__</div>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('backtest')">📊 Backtest Performance</button>
            <button class="tab-btn" onclick="switchTab('live-pnl')">⚡ Forward Paper P&L</button>
            <button class="tab-btn" onclick="switchTab('model-health')">🧠 Model Health</button>
        </div>

        <!-- Backtest Content -->
        <div id="backtest-tab" class="tab-content active">
            <div class="chart-container">
                <div class="chart-header">Symbol Return Comparison</div>
                __BACKTEST_CHART__
            </div>
            <h3>Per-Symbol Metrics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Total Return</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>Active Win Rate</th>
                        <th>Orders Count</th>
                        <th>Active Days</th>
                    </tr>
                </thead>
                <tbody>
                    __BACKTEST_TABLE_ROWS__
                </tbody>
            </table>
        </div>

        <!-- Live P&L Content -->
        <div id="live-pnl-tab" class="tab-content">
            <div class="grid-cards" style="grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));">
                <div class="card">
                    <div class="card-title">Paper Realized Metrics</div>
                    <p><strong>Total Trades:</strong> __LIVE_TRADES_COUNT__</p>
                    <p><strong>Wins / Losses:</strong> __LIVE_WINS__ / __LIVE_LOSSES__</p>
                    <p><strong>Win Rate:</strong> __LIVE_WIN_RATE__</p>
                    <p><strong>Max Drawdown (USD):</strong> __LIVE_MAX_DD__</p>
                </div>
                <div class="card">
                    <div class="card-title">Exclusion reasons</div>
                    __EXCLUSION_REASONS__
                </div>
            </div>
            <h3>Open Positions</h3>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Quantity</th>
                        <th>Entry Price</th>
                        <th>Entry Date</th>
                        <th>Initial Stop Price</th>
                    </tr>
                </thead>
                <tbody>
                    __OPEN_POS_ROWS__
                </tbody>
            </table>
        </div>

        <!-- Model Health Content -->
        <div id="model-health-tab" class="tab-content">
            <h3>Model Status & Accuracy Scorecard</h3>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>RF Accuracy</th>
                        <th>RF F1 Score</th>
                        <th>Out-Of-Sample AUC</th>
                        <th>Holdout AUC</th>
                        <th>Positive Rate</th>
                        <th>Total Samples</th>
                    </tr>
                </thead>
                <tbody>
                    __MODEL_HEALTH_ROWS__
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            event.currentTarget.classList.add('active');
            document.getElementById(tabId + '-tab').classList.add('active');
        }
    </script>
</body>
</html>
"""

def generate_dashboard():
    # 1. Load data
    backtest_metrics = {}
    backtest_path = config.REPORTS_DIR / "backtest_metrics.json"
    if backtest_path.exists():
        try:
            with backtest_path.open("r", encoding="utf-8") as fh:
                backtest_metrics = json.load(fh)
        except Exception as e:
            logger.warning("Failed to load backtest metrics: %s", e)

    live_pnl = {}
    live_path = config.REPORTS_DIR / "forward_test_metrics.json"
    if live_path.exists():
        try:
            with live_path.open("r", encoding="utf-8") as fh:
                live_pnl = json.load(fh)
        except Exception as e:
            logger.warning("Failed to load live metrics: %s", e)

    model_metrics_data = {}
    model_path = config.REPORTS_DIR / "model_metrics_binary.json"
    if not model_path.exists():
        model_path = config.REPORTS_DIR / "model_metrics_runtime_binary.json"
    if model_path.exists():
        try:
            with model_path.open("r", encoding="utf-8") as fh:
                model_metrics_data = json.load(fh)
        except Exception as e:
            logger.warning("Failed to load model metrics: %s", e)

    # 2. Reconstruct placeholders
    gen_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # Backtest cards
    avg_ret_val = backtest_metrics.get("avg_total_return", 0.0)
    avg_ret = f"{avg_ret_val * 100:+.2f}%"
    avg_ret_class = "pos" if avg_ret_val >= 0 else "neg"
    avg_sharpe = f"{backtest_metrics.get('avg_sharpe', 0.0):.2f}"
    avg_dd = f"{backtest_metrics.get('avg_max_drawdown', 0.0) * 100:+.2f}%"

    # Live P&L cards
    live_pnl_val = live_pnl.get("metrics", {}).get("total_realized_pnl", 0.0)
    live_pnl_str = f"${live_pnl_val:+.2f}"
    live_pnl_class = "pos" if live_pnl_val >= 0 else "neg"

    open_pos = live_pnl.get("open_positions", [])
    open_pos_count = str(len(open_pos))

    # Backtest table
    table_rows = []
    per_sym = backtest_metrics.get("per_symbol", {})

    # Generate interactive SVG chart
    chart_svg = ""
    if per_sym:
        symbols = list(per_sym.keys())
        returns = [m.get("total_return", 0.0) * 100 for m in per_sym.values()]

        # Dimensions
        w, h = 800, 200
        padding = 40
        bar_w = (w - 2*padding) / len(symbols)
        max_r = max(max(abs(r) for r in returns), 1.0)

        chart_svg += f'<svg viewBox="0 0 {w} {h}" style="width: 100%; height: auto; font-size: 12px;" xmlns="http://www.w3.org/2000/svg">\n'
        # Grid line (y = 0)
        zero_y = h / 2
        chart_svg += f'  <line x1="{padding}" y1="{zero_y}" x2="{w - padding}" y2="{zero_y}" stroke="#94a3b8" stroke-dasharray="4" />\n'

        for i, (sym, r) in enumerate(zip(symbols, returns)):
            bar_h = (abs(r) / max_r) * (h/2 - padding)
            x = padding + i * bar_w + bar_w/4
            y = zero_y - bar_h if r >= 0 else zero_y
            color = "var(--success)" if r >= 0 else "var(--danger)"

            # Draw bar
            chart_svg += f'  <rect x="{x}" y="{y}" width="{bar_w/2}" height="{bar_h}" fill="{color}" rx="4" />\n'
            # Text label
            label_y = y - 8 if r >= 0 else y + bar_h + 14
            chart_svg += f'  <text x="{x + bar_w/4}" y="{label_y}" text-anchor="middle" fill="var(--text-main)" font-weight="600">{r:+.1f}%</text>\n'
            # Symbol name
            sym_y = h - 10
            chart_svg += f'  <text x="{x + bar_w/4}" y="{sym_y}" text-anchor="middle" fill="var(--text-sub)">{sym}</text>\n'
        chart_svg += "</svg>"

        for sym, m in per_sym.items():
            table_rows.append(f"""
            <tr>
                <td><strong>{sym}</strong></td>
                <td style="color: {'var(--success)' if m['total_return'] >= 0 else 'var(--danger)'}">{m['total_return']*100:+.2f}%</td>
                <td>{m['sharpe_ratio']:.2f}</td>
                <td style="color: var(--danger)">{m['max_drawdown']*100:+.2f}%</td>
                <td>{m['win_rate_active_days']*100:.1f}%</td>
                <td>{m['n_orders']}</td>
                <td>{m['n_active_days']}</td>
            </tr>
            """)
    backtest_table_rows = "\n".join(table_rows)

    # Live P&L metrics
    live_metrics = live_pnl.get("metrics", {})
    live_trades_count = str(live_metrics.get("n_trades_included", 0))
    live_wins = str(live_metrics.get("wins", 0))
    live_losses = str(live_metrics.get("losses", 0))
    wr = live_metrics.get("win_rate")
    live_win_rate = f"{wr * 100:.1f}%" if wr is not None else "n/a"

    live_max_dd_val = live_pnl.get("drawdown", {}).get("max_drawdown_usd", 0.0)
    live_max_dd = f"${live_max_dd_val:.2f}"

    # Exclusion reasons
    excl = live_metrics.get("excluded_by_reason", {})
    if excl:
        excl_list = "".join(f"<p><strong>{k}:</strong> {v}</p>" for k, v in excl.items())
    else:
        excl_list = "<p>No exclusions.</p>"

    # Open position rows
    open_rows = []
    for pos in open_pos:
        # Check if stop price exists from paper ledger
        stop_price = "n/a"
        open_rows.append(f"""
        <tr>
            <td><strong>{pos.get('symbol')}</strong></td>
            <td>{pos.get('qty')}</td>
            <td>${pos.get('entry_price'):.2f}</td>
            <td>{pos.get('entry_date')}</td>
            <td>{stop_price}</td>
        </tr>
        """)
    open_pos_rows = "\n".join(open_rows) if open_rows else "<tr><td colspan='5' style='text-align: center; color: var(--text-sub);'>No open positions.</td></tr>"

    # Model Health
    model_rows = []
    rf_metrics = model_metrics_data.get("rf", {})
    for sym, m in rf_metrics.items():
        # Clean status tag
        status = "healthy"
        model_rows.append(f"""
        <tr>
            <td><strong>{sym}</strong> <span class="status-badge {status}">{status}</span></td>
            <td>{m.get('test_acc', 0.0)*100:.1f}%</td>
            <td>{m.get('test_f1', 0.0):.2f}</td>
            <td>{m.get('auc', 0.0):.2f}</td>
            <td>{m.get('holdout_auc', 0.0):.2f}</td>
            <td>{m.get('positive_rate', 0.0)*100:.1f}%</td>
            <td>{m.get('n_samples', 0)}</td>
        </tr>
        """)
    model_health_rows = "\n".join(model_rows) if model_rows else "<tr><td colspan='7' style='text-align: center; color: var(--text-sub);'>No trained model metrics found.</td></tr>"

    # Compile HTML
    output = HTML_TEMPLATE
    output = output.replace("__GEN_TIME__", gen_time)
    output = output.replace("__AVG_RET__", avg_ret)
    output = output.replace("__AVG_RET_CLASS__", avg_ret_class)
    output = output.replace("__AVG_SHARPE__", avg_sharpe)
    output = output.replace("__AVG_DD__", avg_dd)
    output = output.replace("__LIVE_PNL__", live_pnl_str)
    output = output.replace("__LIVE_PNL_CLASS__", live_pnl_class)
    output = output.replace("__OPEN_POS_COUNT__", open_pos_count)
    output = output.replace("__BACKTEST_CHART__", chart_svg)
    output = output.replace("__BACKTEST_TABLE_ROWS__", backtest_table_rows)
    output = output.replace("__LIVE_TRADES_COUNT__", live_trades_count)
    output = output.replace("__LIVE_WINS__", live_wins)
    output = output.replace("__LIVE_LOSSES__", live_losses)
    output = output.replace("__LIVE_WIN_RATE__", live_win_rate)
    output = output.replace("__LIVE_MAX_DD__", live_max_dd)
    output = output.replace("__EXCLUSION_REASONS__", excl_list)
    output = output.replace("__OPEN_POS_ROWS__", open_pos_rows)
    output = output.replace("__MODEL_HEALTH_ROWS__", model_health_rows)

    # Save to disk
    out_file = config.REPORTS_DIR / "dashboard.html"
    out_file.write_text(output, encoding="utf-8")
    print(f"✓ Dashboard compiled successfully at {out_file}")
    return out_file

if __name__ == '__main__':
    generate_dashboard()
