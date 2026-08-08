# IndexEdge — Index Replication & Smart-Beta Construction
S&amp;P 500 replication and smart-beta construction. Multi-objective optimized sampling, ML-based tracking, regime-conditional smart-beta, and liquidity-aware, bootstrap-validated backtests, benchmarked on tracking error, turnover, and real cost-adjusted alpha. CPU-only.

## Step 9 — Risk layer

The risk layer uses one annualized active-return tracking-error definition across replication,
smart-beta, and bootstrap results. `risk.attribution` implements an exactly reconciling
Brinson–Fachler decomposition of active return into sector allocation, security selection, and
interaction effects, reports uncovered weights rather than silently assigning missing sector data,
and measures the portfolio's composite-factor exposure relative to the benchmark. Regime-conditional
effects use the same calm/normal/volatile classification implemented in `regime`.

The sticky kill-switch trips when annualized tracking error exceeds 5% or when relative wealth
(portfolio value divided by benchmark value) suffers a drawdown greater than 10%. Historical checks
use the maximum observed relative drawdown, including a loss from the initial 1.0 high-water mark;
the switch remains tripped until an explicit manual reset.

Run the complete real-data report with:

```bash
python -m risk.run_risk_layer
```

The limits are disclosed operating conventions, not values fitted to the backtest. Attribution is
renormalized over symbols with both a known sector and period return, so excluded coverage must be
reviewed alongside the decomposition.

## Step 10 — Results and honest comparison

`results.comparison` produces a normalized strategy × AUM × regime table containing name count,
one-way turnover, gross and cost-adjusted return, Sharpe ratio, tracking error, and block-bootstrap
confidence intervals. Sampling methods retain their target and realized name counts in a separate
out-of-sample comparison because crossing sampling configurations with smart-beta variants would
describe portfolios that were never constructed.

Run `python -m results.run_full_comparison` to write `results/output/full_results.csv`,
`sampling_comparison.csv`, and `findings.txt`. Findings are derived from the result tables: they state
whether the best smart-beta portfolio beats full replication after costs at each AUM, whether min-vol
outperforms specifically in volatile regimes, and how LASSO differs from direct optimization at each
name-count target. Negative and inconclusive outcomes are retained rather than replaced with a fixed
narrative.
