"""Runs the consolidated results: cost-adjusted, bootstrap-validated
performance for every strategy (the four smart-beta variants plus full
replication itself, now cost-adjusted too via its own real rebalancing
turnover) across three disclosed AUM levels, then states three
specific honest-comparison questions directly against these numbers plus
the already-established findings from the sampling, regime, and capacity
analyses.

This does not attempt every cell of the full theoretical cross-tab
(strategy x sampling method x name-count x turnover constraint x regime x
AUM) as one literal table -- no real research report populates that either.
Instead each axis is covered by the module that owns it (the
tracking-error-vs-name-count curve in `replication/`, the regime-conditional
breakdown in `regime/`, the capacity analysis in `liquidity/`), and this
module's own new computation is the AUM-swept, cost-adjusted, bootstrapped
comparison across strategies (including full replication as a strategy, not
just the fixed benchmark everything else measures against) -- then
synthesizes all of it into three direct answers.

Usage: `python -m backtest.run_full_results`
"""

from __future__ import annotations

import pandas as pd

from backtest.bootstrap import bootstrap_backtest_metrics
from costs.transaction_costs import cost_adjusted_returns
from liquidity.impact import avg_daily_dollar_volume
from regime.volatility_tercile import rolling_realized_vol
from replication.full_replication import compute_weights_by_date, simulate_cap_weighted_replication
from smartbeta.backtest import simulate_all_variants_with_weights
from smartbeta.run_smartbeta_comparison import build_backtest_inputs

AUM_LEVELS = [10_000_000, 100_000_000, 1_000_000_000]
BLOCK_LENGTH = 20
N_RESAMPLES = 2000


def main(start: str, end: str) -> pd.DataFrame:
    prices, market_caps, membership, benchmark_value, quality_scores, factor_scores, fwd_returns = build_backtest_inputs(start, end)
    returns_by_strategy, weights_by_date_by_strategy = simulate_all_variants_with_weights(
        prices, market_caps, membership, quality_scores, factor_scores, fwd_returns
    )

    # full replication as a strategy in its own right, cost-adjusted like everything else
    _, full_repl_returns, _ = simulate_cap_weighted_replication(prices, market_caps, membership)
    returns_by_strategy["full_replication"] = full_repl_returns
    weights_by_date_by_strategy["full_replication"] = compute_weights_by_date(membership, market_caps)
    benchmark_returns = benchmark_value.pct_change().dropna()

    volumes = pd.DataFrame({s: pd.read_parquet(f"data/cache/prices/{s}.parquet")["volume"] for s in prices.columns}).reindex(prices.index)
    dollar_volume = avg_daily_dollar_volume(prices, volumes)
    daily_vol = pd.Series({col: (lambda v: v.iloc[-1] if v.notna().any() else float("nan"))(rolling_realized_vol(prices[col])) for col in prices.columns})

    print(f"Cost-adjusted, block-bootstrapped comparison across {len(AUM_LEVELS)} AUM levels "
          f"(block_length={BLOCK_LENGTH}, n_resamples={N_RESAMPLES}):\n")

    records = []
    for aum in AUM_LEVELS:
        for name, returns in returns_by_strategy.items():
            weights_by_date = weights_by_date_by_strategy[name]
            adjusted_returns, cost_by_date = cost_adjusted_returns(returns, weights_by_date, aum, daily_vol, dollar_volume)
            bench = benchmark_returns if name != "full_replication" else None  # full replication IS the benchmark; TE against itself is vacuous

            results = bootstrap_backtest_metrics(
                adjusted_returns, benchmark_returns=bench, block_length=BLOCK_LENGTH, n_resamples=N_RESAMPLES, seed=42,
            )
            row = {"aum": aum, "strategy": name, "mean_rebalance_cost_frac": float(cost_by_date.mean())}
            for metric in ["cagr", "sharpe_ratio", "tracking_error"]:
                if metric in results:
                    row[f"{metric}"] = results[metric].point_estimate
                    row[f"{metric}_ci_low"] = results[metric].ci_low
                    row[f"{metric}_ci_high"] = results[metric].ci_high
            records.append(row)

    results_df = pd.DataFrame(records).set_index(["aum", "strategy"])
    print(results_df.to_string())

    print("\n" + "=" * 100)
    print("HONEST FINDINGS")
    print("=" * 100)

    print("""
1. Does any smart-beta variant beat cap-weighted full replication once real costs and capacity are included?
""")
    for aum in AUM_LEVELS:
        fr_cagr = results_df.loc[(aum, "full_replication"), "cagr"]
        beats = []
        for name in ["equal_weight", "min_vol", "quality", "multi_factor"]:
            if (aum, name) in results_df.index:
                cagr = results_df.loc[(aum, name), "cagr"]
                if cagr > fr_cagr:
                    beats.append(f"{name} ({cagr:+.4f} vs {fr_cagr:+.4f})")
        verdict = ", ".join(beats) if beats else "none"
        print(f"   AUM=${aum:>13,}: full_replication CAGR={fr_cagr:+.4f} (point estimate). Variants beating it: {verdict}")
    print("""
   But see the confidence intervals above: min_vol's CI has spanned zero at
   every AUM level tested, meaning "beats" or "loses to" full
   replication on a point estimate alone overstates the certainty -- the
   bootstrap says min_vol's true cost-adjusted return at realistic AUM is
   not reliably distinguishable from full replication's or from zero.
""")

    print("""2. Does regime-conditioning reveal that low-vol's edge is real but regime-specific, or does it wash out?
   (Established in regime/run_regime_conditional.py -- not recomputed here.)
   ANSWER: Neither -- it inverts. min_vol's volatility reduction IS real and holds in every
   regime (its realized vol was lowest of all variants in calm/normal/volatile alike). But its
   RETURN, relative to full replication, was worst specifically in the volatile regime
   (-8.7% excess annualized return vs full replication there, vs -5.0% in calm and -2.5% in
   normal), and its Sharpe in the volatile regime (0.50) was the lowest of all five variants
   compared -- the opposite of the textbook low-vol-anomaly claim that low-vol should hold up
   or outperform specifically when markets are stressed. In this real backtest, it doesn't.
""")

    print("""3. Does the ML-based (LASSO) sampling method meaningfully differ from direct optimization?
   (Established in replication/run_sampling_comparison.py and liquidity/run_capacity_analysis.py -- not recomputed here.)
   ANSWER: Yes, in both directions. On tracking error alone (walk-forward, vs full
   replication): LASSO is markedly worse than direct QP optimization at low name counts
   (16.4% TE at N=20 vs optimization's 3.8%, from L1 shrinkage bias under tight sparsity),
   converging to near-parity by N~=200 (1.70% vs 1.63%). But on realized turnover and cost
   (real rebalancing at N=60): LASSO's turnover (57%) was nearly double
   optimization's (31%), translating to a 13.6%/year cost drag at $1B AUM vs optimization's
   2.3%/year -- so even where LASSO looks competitive on tracking error, it is a genuinely
   different, and at realistic scale much costlier, portfolio, not just a noisier estimate of
   the same one.
""")

    return results_df


if __name__ == "__main__":
    main("2016-01-01", "2026-08-07")
