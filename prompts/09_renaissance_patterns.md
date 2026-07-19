# 9. The Renaissance Technologies Pattern Finder

> Generated: {{generated_at}}
> Stock Engine Pro v3 — research prompt pack

**Ticker:** {{ticker}}
**Time period:** {{time_period}}
**Mode:** {{mode}}

---

## Data context (auto-filled by stock_prompts.py)

- Bars analyzed: {{bar_count}}
- Date range: {{date_range}}
- Avg daily return: {{avg_daily_return}}
- Daily return stdev: {{daily_return_stdev}}
- Avg up-day vs down-day: {{up_down_day_ratio}}
- Days since last earnings: {{days_since_earnings}}

---

"You are a quantitative researcher at Renaissance Technologies using data-driven methods to find statistical edges in the stock market.

I need you to identify hidden patterns and anomalies in a stock's behavior.

Research:

- Seasonal patterns: best and worst months historically
- Day-of-week performance patterns if any exist
- Correlation with major market events (Fed meetings, CPI reports)
- Insider buying and selling patterns from recent filings
- Institutional ownership trend: are big funds buying or selling
- Short interest analysis and squeeze potential
- Unusual options activity signals worth watching
- Price behavior around earnings (pre-run, post-gap patterns)
- Sector rotation signals that affect this stock
- Statistical edge summary: what gives this stock a quantifiable advantage

Format as a quantitative research memo with data tables and pattern summaries.

The stock to investigate: {{ticker}} over {{time_period}}"
