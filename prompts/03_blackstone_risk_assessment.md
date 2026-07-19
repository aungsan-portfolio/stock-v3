# 3. A Blackstone–Style Risk Assessment

> Generated: {{generated_at}}
> Stock Engine Pro v3 — research prompt pack

**Ticker:** {{ticker}}
**Company:** {{company_name}}
**Position size (USD):** {{position_size}}
**Time horizon:** {{time_horizon}}

---

## Data context (auto-filled by stock_prompts.py)

- Annualized volatility (60d): {{vol_60d}}
- Annualized volatility (1y): {{vol_1y}}
- Max drawdown (1y): {{max_drawdown_1y}}
- Beta vs SPY: {{beta}}
- Short interest % of float: {{short_interest_pct}}
- Debt / Equity: {{debt_to_equity}}
- Interest coverage (EBIT / interest): {{interest_coverage}}

---

"You are a managing director in Blackstone's private equity risk practice evaluating potential investments.

I need a thorough risk assessment for a stock I am considering.

Deliver:

- Volatility analysis (30/60/90-day, annualized)
- Beta and correlation with S&P 500
- Value at Risk (VaR) at 95% and 99% confidence intervals
- Maximum drawdown analysis over 1, 3, and 5 years
- Liquidity risk: average daily volume, days-to-exit
- Concentration risk if this becomes 10%+ of my portfolio
- Tail risk scenarios: black swan events, sector collapse
- Stress test: what happens in a 2008-style or 2020-style crash
- Risk score 1–10 with detailed reasoning
- Position sizing recommendation using Kelly Criterion or similar

Format as a risk report with quantitative tables and scenario analysis.

The stock: {{ticker_company}}"
