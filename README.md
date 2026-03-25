# Meridian - QuantConnect Algorithm

A live-tradeable implementation of the Meridian factor model, built natively on QuantConnect's LEAN engine using Morningstar fundamental data. Scores a universe of 500 US equities weekly across value, quality, and momentum factors, then goes long the top quintile and short the bottom quintile.

---

## What this is

The standalone Meridian project (in the parent repo) is a research tool - it fetches data from FMP, scores stocks, and produces reports. This QuantConnect implementation takes the same scoring logic and turns it into a live algorithm that can actually trade.

The key difference is data source. Instead of calling FMP's API, the algorithm uses QuantConnect's built-in Morningstar fundamental database, which covers 8,000+ US equities going back to 1998 and is available for free on the platform. This removes the FMP data constraint entirely for backtesting purposes.

---

## File structure

```
MeridianAlgo/
├── main.py        # Algorithm entry point - initialisation, scheduling, execution
├── factors.py     # Factor definitions and fundamental data extraction
├── scorer.py      # Z-scoring, winsorisation, composite score, quintile assignment
├── portfolio.py   # Long-short portfolio construction and position sizing
└── universe.py    # Universe selection filters - coarse and fine screens
```

Each file maps directly to a module in the standalone Meridian project. The logic is identical; only the data access layer changes.

---

## How it works end to end

### 1. Universe selection (universe.py)

Every day, QuantConnect passes the algorithm a list of all tradeable US equities. The coarse filter runs first and narrows this down to stocks with fundamental data available, a price above $5, and daily dollar volume above $10 million. The top 500 by dollar volume are kept.

The fine filter then runs on those 500 and applies a second screen: excludes financial sector stocks (banks, insurers) because their ratios behave differently from industrials and tech, and requires non-zero fundamental data to be present.

The result is a clean, liquid universe of roughly 400–450 stocks each week.

### 2. Factor extraction (factors.py)

For each stock in the universe, eight fundamental factors are extracted from Morningstar data:

| Factor | Source | What it measures |
|---|---|---|
| `pe_ratio` | ValuationRatios | Price relative to earnings |
| `ev_ebitda` | ValuationRatios | Enterprise value relative to operating profit |
| `pb_ratio` | ValuationRatios | Price relative to book value |
| `fcf_yield` | ValuationRatios | Free cash flow relative to enterprise value |
| `roe` | OperationRatios | Return on equity - profitability per £ of shareholder capital |
| `roic` | OperationRatios | Return on invested capital - capital efficiency |
| `gross_margin` | OperationRatios | Gross profit as a percentage of revenue |
| `debt_to_equity` | OperationRatios | Financial leverage - how much debt vs equity |

Momentum (12-1 month price return) is computed separately using QuantConnect's `History()` call, which pulls daily price data going back 252 trading days.

**Important - OperationRatios are MultiPeriodField objects.** Unlike ValuationRatios which return plain numbers, OperationRatios fields like ROE and ROIC return objects with multiple time periods (.OneYear, .ThreeMonths, etc.). The code calls `.OneYear` on these to get the trailing annual figure.

### 3. Scoring (scorer.py)

The raw factor values go through four transformations before becoming a score:

**Winsorisation** - extreme outlier values are clipped to the 5th–95th percentile range. This prevents a single stock with an absurd P/E ratio (e.g. a company barely breaking even) from distorting every other stock's z-score.

**Cross-sectional z-scoring** - each factor is standardised so the average stock scores 0 and one standard deviation above average scores 1. This puts all factors on the same scale regardless of their units.

**Direction adjustment** - for factors where a lower number is better (P/E, EV/EBITDA, P/B, debt/equity), the sign is flipped. After this step, a higher z-score always means "more attractive" regardless of which factor you're looking at.

**Composite score** - z-scores are averaged within three groups (value, quality, momentum) and then combined using configurable group weights. The default is 40% value, 40% quality, 20% momentum - derived from the IC analysis in the standalone Meridian notebook.

The universe is then ranked by composite score and divided into five equal buckets (quintiles). Q5 is the most attractive, Q1 is the least.

### 4. Portfolio construction (portfolio.py)

Q5 stocks form the long leg. Q1 stocks form the short leg. Each leg targets 50% of the portfolio, so the algorithm is market-neutral by design - roughly $500k long and $500k short on a $1m account.

Position sizing is equal-weighted within each leg, capped at 5% per stock. This means you need at least 10 stocks in each leg to be fully invested, which is easily achieved with a 500-stock universe.

### 5. Execution (main.py)

The rebalance runs every Monday at 30 minutes after market open. This timing avoids the opening auction noise while still executing early in the week.

Before placing any trade, three checks run on each target symbol:
- `ContainsKey` - is the symbol in the current subscription list?
- `IsTradable` - is the security currently active (not delisted or halted)?
- `HasData` and `Price > 0` - has at least one bar of price data been received?

Symbols failing any check are skipped silently. The previous week's positions that are no longer in the signal are exited automatically via the `liquidate=True` flag on `SetHoldings`.

---

## Running a backtest

1. Create a free account at [quantconnect.com](https://quantconnect.com)
2. New Algorithm → blank Python project
3. Create five files (`main.py`, `factors.py`, `scorer.py`, `portfolio.py`, `universe.py`) and paste the contents of each
4. Click Backtest

The default configuration runs 2020–2024 at weekly rebalancing. The first few weeks will be quiet while the warmup period (400 days) completes - this is normal and ensures the algorithm has sufficient price history to compute momentum before trading.

**Expected backtest runtime:** 5–15 minutes depending on QuantConnect server load.

---

## Configuration

All the key parameters live at the top of `main.py` and `factors.py` and can be changed without touching the core logic:

```python
# main.py
self.SetStartDate(2020, 1, 1)       # backtest start
self.SetEndDate(2024, 12, 31)       # backtest end
self.SetCash(1_000_000)             # starting capital

# Rebalance frequency - change DayOfWeek or switch to DateRules.MonthStart
self.Schedule.On(
    self.DateRules.Every(DayOfWeek.Monday),
    self.TimeRules.AfterMarketOpen("SPY", 30),
    self.Rebalance
)
```

```python
# factors.py
GROUP_WEIGHTS = {
    "value":    0.40,
    "quality":  0.40,
    "momentum": 0.20,
}
```

```python
# portfolio.py - in build_targets()
max_position = 0.05    # max 5% per stock
# Long leg targets 50% gross, short leg targets 50% gross
```

---

## Key design decisions

**Why exclude financials?** Banks and insurers use debt as an input to their business model, not just as a funding tool. Debt/equity ratios of 10x are normal for a bank but a red flag for a manufacturer. Including them without sector-specific factor adjustments would produce misleading signals.

**Why weekly rebalancing?** Monthly is too slow for momentum signals to remain fresh; daily is too expensive in transaction costs for a fundamental model where the underlying data only updates monthly. Weekly is the standard for systematic fundamental strategies.

**Why 500 stocks in the universe?** With quintile scoring you need enough names to have statistically meaningful buckets. 500 stocks gives roughly 100 per quintile, meaning the long and short legs each hold ~20 stocks after the position size cap - a reasonable level of diversification without over-diluting the signal.

**Why long-short instead of long-only?** A long-short portfolio is approximately market-neutral, meaning its returns should be driven by the quality of the factor signal rather than general market direction. This makes the Sharpe ratio a cleaner measure of the model's actual alpha. Long-only returns are heavily contaminated by beta.

---

## Relationship to standalone Meridian

| | Standalone Meridian | QuantConnect algo |
|---|---|---|
| Data source | FMP API (paid) | Morningstar via QC (free) |
| Universe | Hand-picked, ~30 stocks | Screened, ~500 stocks |
| Execution | None - signals only | Full live trading via LEAN |
| Backtesting | 2 periods (data limited) | 2020–present (20yr available) |
| Feature selection | IC analysis notebook | Uses weights from notebook |
| European coverage | ADRs via FMP | US only (Morningstar) |

The two implementations are complementary. The standalone Meridian project is where you develop and refine the model - new factors, weight optimisation, IC analysis. The QuantConnect implementation is where you deploy and trade it.

---

## What to look for in the backtest results

**Equity curve** - should be broadly upward-sloping with drawdowns contained. A long-short factor model should not have a drawdown exceeding 25–30% unless it's deeply wrong about something structural.

**Sharpe ratio** - anything above 0.8 on a long-short book is respectable for a pure factor model. Above 1.5 warrants careful examination for data snooping.

**Benchmark comparison** - the algorithm is benchmarked against SPY. In a strong momentum market (2020–2021, 2023–2024) the long-short may underperform SPY's total return because the short leg can be a headwind. The right comparison is absolute Sharpe, not relative return.

**Long leg vs short leg attribution** - the backtest logs print long and short counts each week. If the short leg is consistently generating large losses, the quality or value factors may be misfiring and the GROUP_WEIGHTS should be revisited using the feature selection notebook.

---

## Live trading

QuantConnect supports live trading via Interactive Brokers, Tradier, and other brokers. To deploy this algorithm live:

1. Connect a brokerage account in the QuantConnect dashboard
2. Run the backtest to confirm no runtime errors
3. Deploy via Live Trading → select your brokerage

Note that live trading requires a QuantConnect subscription (starts at $20/month). The short leg also requires margin - confirm your broker account type supports short selling before deploying.