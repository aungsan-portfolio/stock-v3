"""
performance_tracker.py -- Trade expectancy and win/loss ratio tracker.
"""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TradeResult:
    symbol: str
    side: str            # 'LONG' | 'SHORT'
    regime: str          # 'RISK_ON' | 'NEUTRAL' | 'RISK_OFF'
    pnl: float           # realized P&L in USD
    is_counter_trend: bool


def calculate_expectancy(trades: List[TradeResult]) -> Dict[str, Any]:
    """
    Expectancy = (Win Rate * Avg Win) + (Loss Rate * Avg Loss)
    Returns a dictionary summarizing performance metrics.
    """
    if not trades:
        return {}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]

    win_rate = len(wins) / len(trades)
    loss_rate = len(losses) / len(trades)

    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0

    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    risk_on_trades = [t for t in trades if t.regime == "RISK_ON"]
    counter_trend = [t for t in trades if t.is_counter_trend]
    ct_win_rate = (
        len([t for t in counter_trend if t.pnl > 0]) / len(counter_trend)
        if counter_trend
        else 0.0
    )

    return {
        "total_trades": len(trades),
        "win_rate": f"{win_rate:.1%}",
        "loss_rate": f"{loss_rate:.1%}",
        "avg_win": f"${avg_win:.2f}",
        "avg_loss": f"${avg_loss:.2f}",
        "win_loss_ratio": f"{win_loss_ratio:.2f}x",
        "expectancy_per_trade": f"${expectancy:.2f}",
        "risk_on_trade_count": len(risk_on_trades),
        "counter_trend_%": f"{len(counter_trend) / len(trades):.1%}",
        "counter_trend_win_rate": f"{ct_win_rate:.1%}",
    }
