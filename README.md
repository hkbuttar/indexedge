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
