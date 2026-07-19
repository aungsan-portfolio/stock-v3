# 2. A Morgan Stanley–Style DCF Valuation Deep Dive

> Generated: {{generated_at}}
> Stock Engine Pro v3 — research prompt pack

**Ticker:** {{ticker}}
**Company:** {{company_name}}
**Mode:** {{mode}}

---

## Data context (auto-filled by stock_prompts.py)

- Latest revenue (TTM, USD): {{ttm_revenue}}
- Latest EBITDA (TTM, USD): {{ttm_ebitda}}
- Latest free cash flow (TTM, USD): {{ttm_fcf}}
- Shares outstanding (diluted, millions): {{shares_diluted_m}}
- Net debt (USD): {{net_debt}}
- Reference share price (USD): {{reference_price}}
- 5-yr revenue CAGR (historical): {{revenue_cagr_5y}}
- Current beta: {{beta}}

> These inputs seed the WACC and growth assumptions inside the model. Re-run the runner for a refresh.

---

"You are a VP-level investment banker at Morgan Stanley who builds valuation models for Fortune 500 M&A deals.

I need a full discounted cash flow analysis for a specific stock.

Build out:

- 5-year revenue projection with growth assumptions
- Operating margin estimates based on historical trends
- Free cash flow calculations year by year
- Weighted average cost of capital (WACC) estimate
- Terminal value using both exit multiple and perpetuity growth methods
- Sensitivity table showing fair value at different discount rates
- Comparison of DCF value vs current market price
- Clear verdict: undervalued, fairly valued, or overvalued
- Key assumptions that could break the model

Format as an investment banking valuation memo with tables and clear math.

The stock I want valued: {{ticker_company}}"
