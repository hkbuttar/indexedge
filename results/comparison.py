"""Build the consolidated comparison table and derive claims from its data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.bootstrap import bootstrap_backtest_metrics

PERIODS_PER_YEAR = 252


def average_portfolio_metadata(weights_by_date: dict[pd.Timestamp, pd.Series]) -> tuple[float, float]:
    """Return mean holding count and mean one-way turnover across rebalances."""
    if not weights_by_date:
        return float("nan"), float("nan")
    ordered = [weights_by_date[t] for t in sorted(weights_by_date)]
    counts = [int((weights.abs() > 1e-8).sum()) for weights in ordered]
    turnovers = []
    for previous, current in zip(ordered, ordered[1:]):
        universe = previous.index.union(current.index)
        traded = (
            current.reindex(universe).fillna(0.0) - previous.reindex(universe).fillna(0.0)
        ).abs().sum()
        turnovers.append(float(traded) / 2)
    return float(np.mean(counts)), float(np.mean(turnovers)) if turnovers else float("nan")


def build_comparison_table(
    gross_returns: dict[str, pd.Series],
    adjusted_returns_by_aum: dict[float, dict[str, pd.Series]],
    benchmark_returns: pd.Series,
    regime_labels: pd.Series,
    weights_by_strategy: dict[str, dict[pd.Timestamp, pd.Series]],
    *,
    block_length: int = 20,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create strategy × AUM × regime results with bootstrap confidence intervals.

    The ``all`` regime represents the complete history. Each conditional row is
    bootstrapped only from observations actually assigned to that regime.
    """
    records: list[dict[str, object]] = []
    for aum, returns_by_strategy in adjusted_returns_by_aum.items():
        for strategy, adjusted in returns_by_strategy.items():
            mean_names, mean_turnover = average_portfolio_metadata(
                weights_by_strategy.get(strategy, {})
            )
            labels = regime_labels.reindex(adjusted.index)
            subsets = {"all": adjusted}
            subsets.update(
                {
                    str(regime): adjusted[labels == regime]
                    for regime in labels.dropna().unique()
                }
            )
            for regime, subset in subsets.items():
                benchmark_subset = benchmark_returns.reindex(subset.index)
                aligned = pd.concat([subset, benchmark_subset], axis=1).dropna()
                if len(aligned) < block_length:
                    continue
                boot = bootstrap_backtest_metrics(
                    aligned.iloc[:, 0],
                    aligned.iloc[:, 1],
                    block_length=block_length,
                    n_resamples=n_resamples,
                    seed=seed,
                )
                gross = gross_returns[strategy].reindex(aligned.index).dropna()
                gross_ann = float((1 + gross.mean()) ** PERIODS_PER_YEAR - 1)
                records.append(
                    {
                        "strategy": strategy,
                        "strategy_family": (
                            "benchmark" if strategy == "full_replication" else "smart_beta"
                        ),
                        "aum": float(aum),
                        "regime": regime,
                        "name_count": mean_names,
                        "mean_one_way_turnover": mean_turnover,
                        "n_days": len(aligned),
                        "gross_annualized_return": gross_ann,
                        "cost_adjusted_return": boot["cagr"].point_estimate,
                        "cost_adjusted_return_ci_low": boot["cagr"].ci_low,
                        "cost_adjusted_return_ci_high": boot["cagr"].ci_high,
                        "sharpe": boot["sharpe_ratio"].point_estimate,
                        "sharpe_ci_low": boot["sharpe_ratio"].ci_low,
                        "sharpe_ci_high": boot["sharpe_ratio"].ci_high,
                        "tracking_error": boot["tracking_error"].point_estimate,
                        "tracking_error_ci_low": boot["tracking_error"].ci_low,
                        "tracking_error_ci_high": boot["tracking_error"].ci_high,
                    }
                )
    return pd.DataFrame.from_records(records)


def derive_honest_findings(
    comparison: pd.DataFrame, sampling_summary: pd.DataFrame | None = None
) -> list[str]:
    """Generate bounded conclusions, including negative or inconclusive results."""
    findings: list[str] = []
    overall = comparison[comparison["regime"] == "all"]
    for aum, group in overall.groupby("aum"):
        benchmark = group[group["strategy"] == "full_replication"]
        smart_beta = group[group["strategy_family"] == "smart_beta"]
        if benchmark.empty or smart_beta.empty:
            continue
        benchmark_return = float(benchmark.iloc[0]["cost_adjusted_return"])
        winner = smart_beta.sort_values("cost_adjusted_return", ascending=False).iloc[0]
        excess = float(winner["cost_adjusted_return"]) - benchmark_return
        verb = "beat" if excess > 0 else "did not beat"
        findings.append(
            f"At ${aum:,.0f} AUM, {winner['strategy']} {verb} full replication after costs "
            f"({excess:+.2%} annualized difference)."
        )

    volatile = comparison[comparison["regime"] == "volatile"]
    for aum, group in volatile.groupby("aum"):
        minimum_vol = group[group["strategy"] == "min_vol"]
        benchmark = group[group["strategy"] == "full_replication"]
        if minimum_vol.empty or benchmark.empty:
            continue
        excess = float(minimum_vol.iloc[0]["cost_adjusted_return"]) - float(
            benchmark.iloc[0]["cost_adjusted_return"]
        )
        conclusion = "outperformed" if excess > 0 else "underperformed"
        findings.append(
            f"In volatile regimes at ${aum:,.0f} AUM, min_vol {conclusion} full replication "
            f"by {excess:+.2%} annualized; the low-vol claim is therefore regime-specific evidence, "
            "not a universal conclusion."
        )

    if sampling_summary is not None and not sampling_summary.empty:
        for target_n, group in sampling_summary.groupby("target_n"):
            ranked = group.dropna(subset=["mean_tracking_error"]).sort_values("mean_tracking_error")
            if ranked.empty:
                continue
            best = ranked.iloc[0]
            lasso = ranked[ranked["method"] == "lasso"]
            optimized = ranked[ranked["method"] == "optimization"]
            comparison_text = ""
            if not lasso.empty and not optimized.empty:
                delta = float(lasso.iloc[0]["mean_tracking_error"]) - float(
                    optimized.iloc[0]["mean_tracking_error"]
                )
                comparison_text = f"; LASSO minus optimization TE was {delta:+.2%}"
            findings.append(
                f"At target {int(target_n)} names, {best['method']} had the lowest mean "
                f"out-of-sample tracking error ({best['mean_tracking_error']:.2%}){comparison_text}."
            )
    return findings
