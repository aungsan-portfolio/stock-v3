# Stock Engine Pro v3 � Research Prompt Pack

Eight Wall-Street-style analysis templates, ready to drop into ChatGPT, Claude, Gemini, or any LLM. Each template is a plain Markdown file with `{{variable}}` placeholders that the bundled `stock_prompts.py` runner can fill in automatically using live data from this repo (`data_manager`, `market_universe`, `hot_scanner`).

## Templates

| # | File | Persona | Best for |
|---|------|---------|----------|
| 1 | `01_goldman_sachs_screener.md` | Goldman Sachs senior equity analyst | Top-10 stock screen with P/E, D/E, moat, 12m targets |
| 2 | `02_morgan_stanley_dcf.md` | Morgan Stanley VP investment banker | 5y DCF, WACC, sensitivity table, valuation verdict |
| 3 | `03_blackstone_risk_assessment.md` | Blackstone MD private equity risk | VaR, drawdown, stress test, position sizing |
| 4 | `04_jpmorgan_earnings.md` | JPMorgan senior equity research | Pre-earnings brief, bull/bear case, options-implied move |
| 5 | `05_peter_lynch_growth.md` | Peter Lynch (Magellan Fund) | Two-minute drill, category, PEG, Lynch verdict |
| 6 | `06_citadel_technical.md` | Citadel senior quant trader | Multi-TF trend, S/R, indicators, trade plan |
| 7 | `07_harvard_dividend.md` | Harvard endowment CIO | 15-20 dividend picks, payout safety, DRIP compounding |
| 8 | `08_bain_competitive.md` | Bain & Company senior partner | Competitive landscape, moats, SWOT, single best pick |
| 9 | `09_renaissance_patterns.md` | Renaissance Technologies quant | Seasonality, day-of-week, insider, options, edge summary |
| 10 | `10_mckinsey_macro.md` | McKinsey Global Institute senior partner | Rates, CPI, GDP, USD, Fed path, portfolio action plan |

## Usage

### Option A � drop into an LLM by hand

1. Open the template in your editor.
2. Replace the `{{placeholders}}` at the top with your ticker, amounts, etc.
3. Copy the entire `## "You are �` block into your favorite LLM.

### Option B � run via the bundled CLI (recommended)

```powershell
python -X utf8 stock_prompts.py list
python -X utf8 stock_prompts.py show --template 01 --ticker AAPL
python -X utf8 stock_prompts.py render --template 06 --ticker NVDA --position "long 100 @ 850" --out reports\prompts\nvda_technical.md
```

> Note: the runner is the standalone `stock_prompts.py` script (it has its own
> `list` / `show` / `render` subcommands). It is **not** a `main.py` subcommand.

`render` writes a fully filled Markdown brief to `reports\prompts\` that you can paste straight into any LLM. It also calls out (clearly) which data points are auto-filled from this repo vs. which still need a human/LLM answer.

## Data sources

The runner tries to pull live numbers from:

- `data_manager.fetch_ohlcv(symbol)` � price, SMAs, RSI, MACD, ATR, volume
- `market_universe` + `hot_scanner` � candidate counts, sector aggregates
- `config.WATCHLIST` � fallback universe

If a number is not available, the placeholder is left as `TBD` so you can fill it manually.
