# 6. A Citadel–Grade Technical Analysis System

> Generated: {{generated_at}}
> Stock Engine Pro v3 — research prompt pack

**Ticker:** {{ticker}}
**Current position:** {{current_position}}
**Mode:** {{mode}}

---

## Data context (auto-filled by stock_prompts.py)

- Latest close: {{reference_price}}
- 50-day SMA: {{sma50}}
- 100-day SMA: {{sma100}}
- 200-day SMA: {{sma200}}
- 14-day RSI: {{rsi_14}}
- MACD (12/26/9): {{macd_value}}, signal: {{macd_signal}}
- Bollinger %B (20,2): {{bollinger_pct_b}}
- 20d avg volume: {{avg_volume_20d}}
- ATR(14) as % of price: {{atr_pct}}
- 52-week high / low: {{fifty_two_week_range}}

---

"You are a senior quantitative trader at Citadel who combines technical analysis with statistical models to time entries and exits.

I need a full technical analysis breakdown of a stock.
Analyze:

- Current trend direction on daily, weekly, and monthly timeframes
- Key support and resistance levels with exact price points
- Moving average analysis (50-day, 100-day, 200-day) and crossover signals
- RSI, MACD, and Bollinger Band readings with plain-English interpretation
- Volume trend analysis and what it signals about buyer vs seller strength
- Chart pattern identification (head and shoulders, cup and handle, etc.)
- Fibonacci retracement levels for potential bounce zones
- Ideal entry price, stop-loss level, and profit target
- Risk-to-reward ratio for the current setup
- Confidence rating: strong buy, buy, neutral, sell, strong sell

Format as a technical analysis report card with a clear trade plan summary.

The stock to analyze: {{ticker_company}}"
