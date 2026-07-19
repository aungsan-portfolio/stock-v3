# 4. A JPMorgan–Level Earnings Breakdown

> Generated: {{generated_at}}
> Stock Engine Pro v3 — research prompt pack

**Company:** {{company_name}}
**Ticker:** {{ticker}}
**Earnings date:** {{earnings_date}}
**Days until report:** {{days_until_earnings}}

---

## Data context (auto-filled by stock_prompts.py)

- Last 4 reported EPS vs consensus: {{eps_history}}
- Last 4 reported revenue vs consensus: {{revenue_history}}
- Consensus revenue (next qtr, USD): {{consensus_revenue}}
- Consensus EPS (next qtr): {{consensus_eps}}
- Implied move (options market, ±%): {{implied_move_pct}}
- Last 4 post-earnings day moves: {{post_earnings_moves}}
- 5y revenue CAGR: {{revenue_cagr_5y}}

---

"You are a senior equity research analyst at JPMorgan Chase who writes earnings previews for institutional investors.

I need a complete earnings analysis before a company reports.

Deliver:

- Last 4 quarters earnings vs estimates (beat or miss history)
- Revenue and EPS consensus estimates for the upcoming quarter
- Key metrics Wall Street is watching for this specific company
- Segment-by-segment revenue breakdown and trends
- Management guidance from last earnings call summarized
- Options market implied move for earnings day
- Historical stock price reaction after last 4 earnings reports
- Bull case scenario and price impact estimate
- Bear case scenario and downside risk estimate
- My recommended play: buy before, sell before, or wait

Format as a pre-earnings research brief with a decision summary at the top.

The company reporting earnings: {{ticker_company}}"
