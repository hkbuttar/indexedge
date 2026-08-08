"""Run Step 10 and write the consolidated evidence table and findings.

Usage: ``python -m results.run_full_comparison``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from costs.transaction_costs import cost_adjusted_returns
from data.prices import fetch_index_level
from data.wikipedia_constituents import fetch_constituents_and_changes
from liquidity.impact import avg_daily_dollar_volume
from regime.volatility_tercile import classify_regimes, rolling_realized_vol
from replication.sampling_evaluation import evaluate_sampling_methods, summarize_curve
from results.comparison import build_comparison_table, derive_honest_findings
from smartbeta.backtest import simulate_all_variants_with_weights
from smartbeta.run_smartbeta_comparison import build_backtest_inputs

AUM_LEVELS = [10_000_000, 100_000_000, 1_000_000_000]
TARGET_COUNTS = [30, 60, 100]
BLOCK_LENGTH = 20
N_RESAMPLES = 2000
OUTPUT_DIR = Path(__file__).parent / "output"


def main(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    inputs = build_backtest_inputs(start, end)
    prices, market_caps, membership, benchmark_value, quality, factors, forward = inputs
    gross, weights = simulate_all_variants_with_weights(
        prices, market_caps, membership, quality, factors, forward
    )
    benchmark_returns = benchmark_value.pct_change().dropna()
    gross["full_replication"] = benchmark_returns
    weights["full_replication"] = {}

    volumes = pd.DataFrame(
        {
            symbol: pd.read_parquet(f"data/cache/prices/{symbol}.parquet")["volume"]
            for symbol in prices.columns
        }
    ).reindex(prices.index)
    dollar_volume = avg_daily_dollar_volume(prices, volumes)
    daily_volatility = pd.Series(
        {
            symbol: volatility.dropna().iloc[-1] if volatility.notna().any() else float("nan")
            for symbol in prices.columns
            for volatility in [rolling_realized_vol(prices[symbol])]
        }
    )

    adjusted_by_aum: dict[float, dict[str, pd.Series]] = {}
    for aum in AUM_LEVELS:
        adjusted_by_aum[aum] = {"full_replication": benchmark_returns}
        for strategy, returns in gross.items():
            if strategy == "full_replication":
                continue
            adjusted, _ = cost_adjusted_returns(
                returns, weights[strategy], aum, daily_volatility, dollar_volume
            )
            adjusted_by_aum[aum][strategy] = adjusted

    index_level = fetch_index_level("^GSPC")["close"]
    regimes = classify_regimes(rolling_realized_vol(index_level)).labels
    comparison = build_comparison_table(
        gross,
        adjusted_by_aum,
        benchmark_returns,
        regimes,
        weights,
        block_length=BLOCK_LENGTH,
        n_resamples=N_RESAMPLES,
    )

    current, _ = fetch_constituents_and_changes()
    sectors = dict(zip(current["yfinance_symbol"], current["gics_sector"]))
    sampling_detail = evaluate_sampling_methods(
        prices,
        market_caps,
        membership,
        benchmark_value,
        sectors,
        TARGET_COUNTS,
    )
    sampling_summary = summarize_curve(sampling_detail)
    findings = derive_honest_findings(comparison, sampling_summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "full_results.csv", index=False)
    sampling_summary.to_csv(OUTPUT_DIR / "sampling_comparison.csv", index=False)
    (OUTPUT_DIR / "findings.txt").write_text("\n".join(f"- {item}" for item in findings) + "\n")

    print(comparison.to_string(index=False))
    print("\nHonest findings:")
    print("\n".join(f"- {item}" for item in findings))
    return comparison, sampling_summary, findings


if __name__ == "__main__":
    main("2016-01-01", "2026-08-07")
