"""Block bootstrap confidence intervals for backtest performance metrics.

Ported directly from pairtrade-lab-1's `backtest/bootstrap.py` -- same
circular block bootstrap (Politis & Romano, 1992), same defaults
(`block_length=20`, `n_resamples=2000`, `confidence_level=0.95`), same
percentile-CI approach, same disclosed reasoning for why block_length is an
unfitted, disclosed parameter rather than something derived from the data:
too short and it degenerates toward the naive i.i.d. bootstrap it exists to
improve on (destroying real autocorrelation in daily strategy returns);
too long and too few effectively-independent blocks exist to vary across
resamples, understating uncertainty a different way. This is the plan's
explicit ask for Step 8: reuse the statistical-rigor standard established in
BookMaker, ExecEdge, and PairTrade Lab, not invent a fresh one.

One real extension beyond the ported version: IndexEdge's strategies are
compared against a benchmark (tracking error), which pairtrade-lab-1's
pairs-trading metrics have no equivalent of. `block_bootstrap_resample_paired`
draws the SAME block indices for both the strategy and benchmark return
series on every resample, so day-to-day pairing is preserved and tracking
error computed on a resampled path means what it should -- resampling each
series independently would scramble the pairing and fabricate a tracking
error unrelated to the real one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import metrics as backtest_metrics
from risk.tracking_error import annualized_tracking_error

DEFAULT_BLOCK_LENGTH = 20
DEFAULT_N_RESAMPLES = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low


def _circular_block_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    indices = np.concatenate([(start + np.arange(block_length)) % n for start in starts])
    return indices[:n]


def block_bootstrap_resample(values: np.ndarray, block_length: int, n_resamples: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(values)
    resampled = np.empty((n_resamples, n))
    for i in range(n_resamples):
        indices = _circular_block_bootstrap_indices(n, block_length, rng)
        resampled[i] = values[indices]
    return resampled


def block_bootstrap_resample_paired(
    values_by_name: dict[str, np.ndarray], block_length: int, n_resamples: int, seed: int | None = None
) -> dict[str, np.ndarray]:
    """Same block indices applied to every series in `values_by_name` on
    each resample -- required whenever a metric (like tracking error) needs
    two series' pairing preserved, not just each series' own structure."""
    lengths = {len(v) for v in values_by_name.values()}
    if len(lengths) != 1:
        raise ValueError(f"all series must have the same length, got {lengths}")
    n = lengths.pop()

    rng = np.random.default_rng(seed)
    resampled = {name: np.empty((n_resamples, n)) for name in values_by_name}
    for i in range(n_resamples):
        indices = _circular_block_bootstrap_indices(n, block_length, rng)
        for name, arr in values_by_name.items():
            resampled[name][i] = arr[indices]
    return resampled


def _equity_from_returns(returns: pd.Series, starting_equity: float = 1.0) -> pd.Series:
    growth = (1 + returns).cumprod()
    return pd.concat([pd.Series([starting_equity]), starting_equity * growth], ignore_index=True)


def _compute_all_metrics(returns: pd.Series, benchmark_returns: pd.Series | None) -> dict[str, float]:
    equity = _equity_from_returns(returns)
    metrics = {
        "cagr": backtest_metrics.cagr(equity),
        "sharpe_ratio": backtest_metrics.sharpe_ratio(returns),
        "sortino_ratio": backtest_metrics.sortino_ratio(returns),
        "max_drawdown": backtest_metrics.max_drawdown(equity),
        "win_rate": backtest_metrics.win_rate(returns),
    }
    if benchmark_returns is not None:
        metrics["tracking_error"] = annualized_tracking_error(returns, benchmark_returns)
    return metrics


def bootstrap_backtest_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int | None = None,
) -> dict[str, BootstrapResult]:
    """Block bootstrap CIs for CAGR, Sharpe, Sortino, max drawdown, win rate,
    and (if `benchmark_returns` is given) tracking error -- all from one
    shared set of resampled paths, paired with the benchmark where relevant.

    Raises:
        ValueError: if fewer than block_length non-NaN, benchmark-aligned
            return observations are available.
    """
    if benchmark_returns is not None:
        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
        aligned.columns = ["returns", "benchmark"]
        clean, bench_clean = aligned["returns"], aligned["benchmark"]
    else:
        clean, bench_clean = returns.dropna(), None

    if len(clean) < block_length:
        raise ValueError(f"need at least block_length={block_length} return observations, got {len(clean)}")

    point_estimates = _compute_all_metrics(clean, bench_clean)

    if bench_clean is not None:
        resampled = block_bootstrap_resample_paired(
            {"returns": clean.to_numpy(), "benchmark": bench_clean.to_numpy()}, block_length, n_resamples, seed=seed
        )
        resampled_returns, resampled_benchmark = resampled["returns"], resampled["benchmark"]
    else:
        resampled_returns = block_bootstrap_resample(clean.to_numpy(), block_length, n_resamples, seed=seed)
        resampled_benchmark = None

    distributions: dict[str, list[float]] = {name: [] for name in point_estimates}
    for i in range(n_resamples):
        r = pd.Series(resampled_returns[i])
        b = pd.Series(resampled_benchmark[i]) if resampled_benchmark is not None else None
        for name, value in _compute_all_metrics(r, b).items():
            distributions[name].append(value)

    alpha = 1 - confidence_level
    results = {}
    for name, point in point_estimates.items():
        dist = np.array(distributions[name])
        dist = dist[~np.isnan(dist)]
        if len(dist) == 0:
            ci_low, ci_high = float("nan"), float("nan")
        else:
            ci_low, ci_high = (float(q) for q in np.quantile(dist, [alpha / 2, 1 - alpha / 2]))
        results[name] = BootstrapResult(
            point_estimate=point, ci_low=ci_low, ci_high=ci_high,
            n_resamples=n_resamples, confidence_level=confidence_level,
        )

    return results
