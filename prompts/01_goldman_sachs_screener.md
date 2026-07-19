# 1. A Goldman Sachs–Level Stock Screener

> Generated: {{generated_at}}
> Stock Engine Pro v3 — research prompt pack

**Mode:** {{mode}}
**Account type:** {{account_type}}
**Risk tolerance:** {{risk_tolerance}}
**Investment amount (USD):** {{investment_amount}}
**Time horizon:** {{time_horizon}}
**Preferred sectors:** {{preferred_sectors}}
**Excluded sectors:** {{excluded_sectors}}

---

## Data context (auto-filled by stock_prompts.py)

- Universe source: {{universe_source}}
- Candidate count considered: {{candidate_count}}
- Reference price (latest close, USD): {{reference_price}}
- Reference market cap (USD): {{reference_market_cap}}
- 52-week range: {{fifty_two_week_range}}
- Sector / industry: {{sector_industry}}

> Fill in the bracketed inputs and re-run `python -X utf8 main.py prompts screen` to refresh numbers.

---

"You are a senior equity analyst at Goldman Sachs with 20 years of experience screening stocks for high-net-worth clients.

I need a complete stock screening framework for my investment goals.

Analyze and provide:

- Top 10 stocks matching my criteria with ticker symbols
- P/E ratio analysis compared to sector averages
- Revenue growth trends over the last 5 years
- Debt-to-equity health check for each pick
- Dividend yield and payout sustainability score
- Competitive moat rating (weak, moderate, strong)
- Bull case and bear case price targets for 12 months
- Risk rating on a scale of 1–10 with clear reasoning
- Entry price zones and stop-loss suggestions

Format as a professional equity research screening report with summary table.

My investment profile: {{investment_profile}}"
