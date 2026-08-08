"""IndexEdge API: one endpoint per analytical module in this project, each a
thin wrapper over that module's own functions -- no computation lives here
that doesn't already live in the module it exposes. Pattern (route
structure, CORS, `to_jsonable`) reused from riskdesk's `backend/main.py`.

One deliberate deviation from riskdesk's live-recompute-per-request
pattern, disclosed here rather than silently copied: riskdesk's positions
are genuinely live (fetched from Alpaca fresh on every request, since they
can change between requests), so it never caches. IndexEdge's backtest
inputs are a fixed 2016-2026 historical window read from local parquet
cache (`data/cache/`) -- nothing about them changes during this process's
life. `_STATE` is therefore computed ONCE at import time and reused across
requests, not because caching is free of tradeoffs in general, but because
there's no staleness risk here to trade away.

Endpoints still vary in cost, and that's disclosed per-endpoint rather than
hidden behind a uniform "fast API" assumption: `/api/replication/sampling`
recomputes a walk-forward evaluation (~15-20s) live; `/api/results` instead
serves `results/run_full_comparison.py`'s precomputed output
(`results/output/*.csv`, `findings.txt`) because that computation --
2000-resample bootstraps x 3 AUM levels x 5 strategies -- takes well over a
minute, genuinely too slow for a live request, and is exactly the kind of
expensive/rarely-changing analysis production systems serve pre-computed
rather than inline.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.serialize import to_jsonable
from backtest.bootstrap import bootstrap_backtest_metrics
from costs.transaction_costs import cost_adjusted_returns
from data.prices import fetch_index_level
from data.wikipedia_constituents import fetch_constituents_and_changes
from liquidity.capacity import estimate_portfolio_trade_cost
from liquidity.impact import avg_daily_dollar_volume
from regime.conditional import summarize_regime_conditional_performance
from regime.volatility_tercile import classify_regimes, rolling_realized_vol
from replication.full_replication import rebalance_weights, simulate_cap_weighted_replication
from replication.multi_objective import trace_pareto_frontier
from replication.sampling_evaluation import evaluate_sampling_methods, summarize_curve
from risk.attribution import brinson_fachler_attribution, factor_exposure_differential
from risk.kill_switch import KillSwitch, check_relative_drawdown_limit, check_tracking_error_limit
from risk.tracking_error import summarize_tracking
from smartbeta.backtest import simulate_all_variants_with_weights
from smartbeta.multi_factor import composite_score, trailing_ic_weights
from smartbeta.run_smartbeta_comparison import build_backtest_inputs

REPO_ROOT = Path(__file__).parent.parent
RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "output"
BACKTEST_START, BACKTEST_END = "2016-01-01", "2026-08-07"
DEFAULT_AUM = 100_000_000
DEFAULT_LIVE_N_RESAMPLES = 500  # smaller than the 2000 used offline, disclosed tradeoff for request latency


def _load_state():
    prices, market_caps, membership, benchmark_value, quality_scores, factor_scores, fwd_returns = build_backtest_inputs(
        BACKTEST_START, BACKTEST_END
    )
    returns_by_strategy, weights_by_date_by_strategy = simulate_all_variants_with_weights(
        prices, market_caps, membership, quality_scores, factor_scores, fwd_returns
    )
    benchmark_returns = benchmark_value.pct_change().dropna()
    current, changes = fetch_constituents_and_changes()
    sector_by_symbol = dict(zip(current["yfinance_symbol"], current["gics_sector"]))

    # Each symbol's volume series is sliced to prices.index's own (already
    # lookback-trimmed) range before joining, not after: reading and
    # retaining all 611 symbols' FULL history (back to 1962 for many) just
    # to immediately reindex down to ~2015-2026 was a second, separate
    # contributor to the same out-of-memory failure diagnosed in
    # `replication/full_replication.py` and `smartbeta/run_smartbeta_comparison.py`
    # -- trimming per-symbol before combining keeps only one small slice
    # alive at a time instead of 611 full-length Series simultaneously.
    volume_cutoff = prices.index.min()
    volumes = pd.DataFrame({
        s: pd.read_parquet(f"data/cache/prices/{s}.parquet")["volume"].pipe(lambda v: v[v.index >= volume_cutoff])
        for s in prices.columns
    }).reindex(prices.index)
    dollar_volume = avg_daily_dollar_volume(prices, volumes)
    daily_vol = pd.Series({col: (lambda v: v.iloc[-1] if v.notna().any() else float("nan"))(rolling_realized_vol(prices[col])) for col in prices.columns})

    return {
        "prices": prices, "market_caps": market_caps, "membership": membership,
        "benchmark_value": benchmark_value, "benchmark_returns": benchmark_returns,
        "quality_scores": quality_scores, "factor_scores": factor_scores, "fwd_returns": fwd_returns,
        "returns_by_strategy": returns_by_strategy, "weights_by_date_by_strategy": weights_by_date_by_strategy,
        "current": current, "changes": changes, "sector_by_symbol": sector_by_symbol,
        "dollar_volume": dollar_volume, "daily_vol": daily_vol,
    }


_STATE = _load_state()

app = FastAPI(
    title="IndexEdge API",
    description="S&P 500 replication, optimized sampling, smart-beta, regime-conditional, "
                "capacity-aware, and bootstrap-validated results.",
    version="1.0.0",
)

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = [origin.strip() for origin in _allowed_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allowed_origins.strip() == "*" else _origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "indexedge", "version": app.version}


@app.get("/api/replication/full")
def replication_full() -> dict:
    _, returns, coverage = simulate_cap_weighted_replication(
        _STATE["prices"], _STATE["market_caps"], _STATE["membership"]
    )
    price_returns = fetch_index_level("^GSPC")["close"].pct_change().dropna()
    tr_returns = fetch_index_level("^SP500TR")["close"].pct_change().dropna()
    return to_jsonable({
        "vs_price_index": summarize_tracking(returns, price_returns),
        "vs_total_return_index": summarize_tracking(returns, tr_returns),
        "coverage_by_rebalance": [
            {"rebalance_date": c.rebalance_date, "intended_members": c.intended_members,
             "weighted_members": c.weighted_members, "coverage_fraction": c.coverage_fraction}
            for c in coverage
        ],
    })


@app.get("/api/replication/sampling")
def replication_sampling(target_counts: str = Query("30,60,100")) -> dict:
    counts = [int(x) for x in target_counts.split(",")]
    detail = evaluate_sampling_methods(
        _STATE["prices"], _STATE["market_caps"], _STATE["membership"], _STATE["benchmark_value"],
        _STATE["sector_by_symbol"], counts,
    )
    return to_jsonable({"target_counts": counts, "curve": summarize_curve(detail)})


@app.get("/api/multi-objective")
def multi_objective(
    turnover_budgets: str = Query("0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0"),
) -> dict:
    """Traces the tracking-error-vs-turnover Pareto frontier at
    the most recent real rebalance, starting from the multi-factor tilt's
    actual prior holding (reusing `_STATE`'s already-computed weights
    rather than re-deriving them, unlike the standalone
    `replication/run_multi_objective.py` script this mirrors)."""
    budgets = [float(x) for x in turnover_budgets.split(",")]
    weights_by_date = _STATE["weights_by_date_by_strategy"]["multi_factor"]
    dates = sorted(weights_by_date.keys())
    if len(dates) < 2:
        return to_jsonable({"error": "not enough rebalance history to trace a frontier"})
    t_prev, t_curr = dates[-2], dates[-1]
    prev_weights = weights_by_date[t_prev]

    prices, all_returns = _STATE["prices"], _STATE["prices"].pct_change()
    pos = all_returns.index.searchsorted(t_curr)
    if pos < 252:
        return to_jsonable({"error": "not enough trailing history before the latest rebalance"})
    trailing_returns = all_returns.iloc[pos - 252: pos]
    period_start = prices.index[pos]

    members = set(_STATE["membership"].loc[_STATE["membership"]["rebalance_date"] == t_curr, "symbol"])
    ic_weights = trailing_ic_weights(_STATE["factor_scores"], _STATE["fwd_returns"], prices.index, t_curr)
    composite = composite_score(_STATE["factor_scores"], ic_weights, period_start)
    candidates = [s for s in members if s in trailing_returns.columns and trailing_returns[s].notna().all()
                  and s in composite.index and pd.notna(composite[s])]

    cap_row = _STATE["market_caps"].loc[_STATE["market_caps"].index[_STATE["market_caps"].index.searchsorted(t_curr)]]
    bench_candidates = [s for s in members if s in cap_row.index and pd.notna(cap_row[s]) and cap_row[s] > 0]
    bench_weights = cap_row[bench_candidates] / cap_row[bench_candidates].sum()
    benchmark_returns = trailing_returns[bench_candidates].mul(bench_weights, axis=1).sum(axis=1)

    exposure_median = float(composite[candidates].median())
    exposure_high = float(composite[candidates].quantile(0.75))
    factor_targets = [-10.0, exposure_median, exposure_high]

    frontier = trace_pareto_frontier(
        trailing_returns[candidates], benchmark_returns, prev_weights, composite[candidates],
        factor_targets=factor_targets, turnover_budgets=budgets,
    )
    return to_jsonable({
        "rebalance_date": t_curr, "n_candidates": len(candidates),
        "factor_targets": {"unconstrained": factor_targets[0], "median": factor_targets[1], "p75": factor_targets[2]},
        "frontier": frontier,
    })


@app.get("/api/smartbeta")
def smartbeta() -> dict:
    rows = []
    for name, returns in _STATE["returns_by_strategy"].items():
        summary = summarize_tracking(returns, _STATE["benchmark_returns"])
        rows.append({
            "strategy": name,
            "annualized_return": float((1 + returns.mean()) ** 252 - 1),
            "annualized_vol": float(returns.std(ddof=1) * (252 ** 0.5)),
            "tracking_error_vs_full_replication": summary.tracking_error_annualized,
            "correlation": summary.correlation,
        })
    return to_jsonable({"strategies": rows})


@app.get("/api/regime")
def regime() -> dict:
    index_level = fetch_index_level("^GSPC")["close"]
    vol = rolling_realized_vol(index_level)
    regimes = classify_regimes(vol)

    returns_with_benchmark = dict(_STATE["returns_by_strategy"])
    returns_with_benchmark["full_replication"] = _STATE["benchmark_returns"]
    summary = summarize_regime_conditional_performance(returns_with_benchmark, regimes.labels)
    return to_jsonable({
        "regime_day_counts": regimes.value_counts(),
        "current_regime": regimes.current(),
        "performance_by_strategy_and_regime": summary,
    })


@app.get("/api/liquidity")
def liquidity(aum: float = Query(DEFAULT_AUM, gt=0)) -> dict:
    dates = sorted(_STATE["weights_by_date_by_strategy"].get("multi_factor", {}).keys())
    if len(dates) < 2:
        return to_jsonable({"error": "not enough rebalance history to price a trade"})
    t_prev, t_curr = dates[-2], dates[-1]

    rows = []
    for name, weights_by_date in _STATE["weights_by_date_by_strategy"].items():
        if t_curr not in weights_by_date:
            continue
        prev = weights_by_date.get(t_prev, pd.Series(dtype=float))
        cost = estimate_portfolio_trade_cost(weights_by_date[t_curr], prev, aum, _STATE["daily_vol"], _STATE["dollar_volume"])
        turnover = float((weights_by_date[t_curr].reindex(weights_by_date[t_curr].index.union(prev.index)).fillna(0)
                           - prev.reindex(weights_by_date[t_curr].index.union(prev.index)).fillna(0)).abs().sum())
        rows.append({
            "strategy": name, "aum": aum, "rebalance_date": t_curr,
            "n_holdings": int((weights_by_date[t_curr] > 1e-6).sum()), "turnover": turnover,
            "rebalance_cost_fraction": cost.total_cost_fraction, "annualized_cost_drag": cost.total_cost_fraction * 4,
        })
    return to_jsonable({"aum": aum, "rebalance_date": t_curr, "strategies": rows})


@app.get("/api/backtest/bootstrap")
def backtest_bootstrap(
    strategy: str = Query(..., description="equal_weight | min_vol | quality | multi_factor"),
    aum: float = Query(DEFAULT_AUM, gt=0),
    n_resamples: int = Query(DEFAULT_LIVE_N_RESAMPLES, gt=0, le=5000),
) -> dict:
    if strategy not in _STATE["returns_by_strategy"]:
        return to_jsonable({"error": f"unknown strategy {strategy!r}"})

    adjusted_returns, cost_by_date = cost_adjusted_returns(
        _STATE["returns_by_strategy"][strategy], _STATE["weights_by_date_by_strategy"][strategy],
        aum, _STATE["daily_vol"], _STATE["dollar_volume"],
    )
    metrics = bootstrap_backtest_metrics(
        adjusted_returns, benchmark_returns=_STATE["benchmark_returns"], n_resamples=n_resamples, seed=42,
    )
    return to_jsonable({
        "strategy": strategy, "aum": aum, "n_resamples": n_resamples,
        "mean_rebalance_cost_fraction": float(cost_by_date.mean()),
        "metrics": {name: result for name, result in metrics.items()},
    })


@app.get("/api/risk/attribution")
def risk_attribution(strategy: str = Query("multi_factor")) -> dict:
    weights_by_date = _STATE["weights_by_date_by_strategy"].get(strategy, {})
    dates = sorted(weights_by_date.keys())
    if len(dates) < 2:
        return to_jsonable({"error": "not enough rebalance history for an attribution period"})
    t_prev, t_curr = dates[-2], dates[-1]
    portfolio_weights = weights_by_date[t_prev]

    prices = _STATE["prices"]
    pos_prev, pos_curr = prices.index.searchsorted(t_prev), prices.index.searchsorted(t_curr)
    period_start, period_end = prices.index[pos_prev], prices.index[min(pos_curr, len(prices.index) - 1)]
    period_returns = prices.loc[period_end] / prices.loc[period_start] - 1

    members_prev = set(_STATE["membership"].loc[_STATE["membership"]["rebalance_date"] == t_prev, "symbol"])
    cap_row = _STATE["market_caps"].loc[_STATE["market_caps"].index[_STATE["market_caps"].index.searchsorted(t_prev)]]
    benchmark_weights = rebalance_weights(members_prev, cap_row)

    attribution = brinson_fachler_attribution(portfolio_weights, benchmark_weights, period_returns, _STATE["sector_by_symbol"])

    factor_diff = None
    if strategy == "multi_factor":
        ic_weights = trailing_ic_weights(_STATE["factor_scores"], _STATE["fwd_returns"], prices.index, t_prev)
        composite = composite_score(_STATE["factor_scores"], ic_weights, period_start)
        factor_diff = factor_exposure_differential(portfolio_weights, benchmark_weights, composite)

    return to_jsonable({
        "strategy": strategy, "period_start": t_prev, "period_end": t_curr,
        "attribution": attribution, "factor_exposure_differential": factor_diff,
    })


@app.get("/api/risk/kill-switch")
def risk_kill_switch(aum: float = Query(DEFAULT_AUM, gt=0)) -> dict:
    rows = []
    for name, returns in _STATE["returns_by_strategy"].items():
        weights_by_date = _STATE["weights_by_date_by_strategy"][name]
        adjusted_returns, _ = cost_adjusted_returns(returns, weights_by_date, aum, _STATE["daily_vol"], _STATE["dollar_volume"])
        te_check = check_tracking_error_limit(adjusted_returns, _STATE["benchmark_returns"])
        dd_check = check_relative_drawdown_limit(adjusted_returns, _STATE["benchmark_returns"])
        switch = KillSwitch()
        switch.check([te_check, dd_check])
        rows.append({
            "strategy": name, "tracking_error_check": te_check, "relative_drawdown_check": dd_check,
            "triggered": switch.triggered, "trigger_reasons": switch.trigger_reasons,
        })
    return to_jsonable({"aum": aum, "strategies": rows})


@app.get("/api/results")
def results() -> dict:
    """Serves `results/run_full_comparison.py`'s precomputed output --
    the one endpoint that reads from disk instead of computing live. See
    module docstring for why: this specific computation is genuinely too
    expensive (2000-resample bootstraps x 3 AUM levels x 5 strategies) for
    a live request."""
    comparison_path = RESULTS_OUTPUT_DIR / "full_results.csv"
    sampling_path = RESULTS_OUTPUT_DIR / "sampling_comparison.csv"
    findings_path = RESULTS_OUTPUT_DIR / "findings.txt"

    if not comparison_path.exists():
        return to_jsonable({"error": "results/output/full_results.csv not found -- run `python -m results.run_full_comparison` first"})

    comparison = pd.read_csv(comparison_path)
    sampling = pd.read_csv(sampling_path) if sampling_path.exists() else pd.DataFrame()
    # findings.txt's own "- " bullet prefix is for reading the file directly;
    # the frontend renders these as <li> items, which supplies its own bullet.
    findings = [line.removeprefix("- ") for line in findings_path.read_text().splitlines()] if findings_path.exists() else []

    return to_jsonable({"comparison": comparison, "sampling_comparison": sampling, "findings": findings})
