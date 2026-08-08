"""Kill-switch on tracking error or relative-drawdown breach.

Shape reused directly from riskdesk's `monitor/kill_switch.py` -- the
newest and most extended of the four sibling kill-switch implementations
(alpha-signal-lab's original, copied verbatim into pairtrade-lab-1;
bookmaker's absolute-dollar adaptation; execedge's unrelated order-halt
version): a dict-of-independently-checked-breaches `check()` call, sticky
trigger (`triggered` never clears itself -- only an explicit `reset()`
call re-arms it, "a kill-switch should not silently forgive itself," a
design invariant stated in every sibling's docstring), and named
`trigger_reasons` so a human reviewing a trip knows which limit(s) fired,
not just that something did.

What's genuinely new here (no sibling kill-switch is benchmark-relative):
the two limit checks below trigger on *index-relative* quantities, since
this project tracks/tilts against a benchmark rather than managing
absolute risk. `check_relative_drawdown_limit` reuses
`backtest.metrics.running_drawdown`'s exact formula (`1 - value/peak`) but
applied to the RATIO of portfolio value to benchmark value, not raw
portfolio equity -- i.e. how far the strategy has fallen from its
best-ever relative standing versus the benchmark, not how far the
portfolio itself is down.

Thresholds are disclosed conventions, not fitted to this project's own
results: `TE_LIMIT=0.05` (5% annualized) is a common industry rule-of-thumb
tracking-error budget for a "moderate active risk" enhanced-index/smart-beta
product (materially tighter than an unconstrained active fund, materially
looser than a pure index fund). `RELATIVE_DRAWDOWN_LIMIT=0.10` is set
tighter than the sibling kill-switches' 15% *absolute* portfolio-drawdown
default, since relative (index-hugging) drawdowns are typically smaller in
magnitude than absolute ones for a benchmark-anchored strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from risk.tracking_error import annualized_tracking_error

TE_LIMIT = 0.05
RELATIVE_DRAWDOWN_LIMIT = 0.10


@dataclass
class LimitCheckResult:
    name: str
    breached: bool
    detail: str


def check_tracking_error_limit(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series, limit: float = TE_LIMIT
) -> LimitCheckResult:
    if limit < 0:
        raise ValueError("tracking-error limit must be non-negative")
    te = annualized_tracking_error(portfolio_returns, benchmark_returns)
    breached = bool(te > limit) if te == te else False  # te != te -> NaN, treated as not breached
    return LimitCheckResult(
        name="tracking_error",
        breached=breached,
        detail=f"annualized TE={te:.4f} vs limit={limit:.4f}",
    )


def relative_value_series(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    """Cumulative portfolio value / cumulative benchmark value, both
    normalized to 1.0 at the start of the aligned window -- the series
    `check_relative_drawdown_limit` runs `running_drawdown` over."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["portfolio", "benchmark"]
    if (aligned <= -1).any().any():
        raise ValueError("returns must be greater than -100%")
    portfolio_value = (1 + aligned["portfolio"]).cumprod()
    benchmark_value = (1 + aligned["benchmark"]).cumprod()
    return portfolio_value / benchmark_value


def check_relative_drawdown_limit(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series, limit: float = RELATIVE_DRAWDOWN_LIMIT
) -> LimitCheckResult:
    if not 0 <= limit < 1:
        raise ValueError("relative-drawdown limit must be in [0, 1)")
    relative_value = relative_value_series(portfolio_returns, benchmark_returns)
    if relative_value.empty:
        return LimitCheckResult(name="relative_drawdown", breached=False, detail="no overlapping data")

    # Include the initial 1.0 high-water mark. Without it, an immediate
    # relative loss on the first observation is incorrectly treated as a peak.
    running_peak = relative_value.cummax().clip(lower=1.0)
    drawdown = 1 - relative_value / running_peak
    max_drawdown = float(drawdown.max())
    breached = max_drawdown > limit
    return LimitCheckResult(
        name="relative_drawdown",
        breached=breached,
        detail=f"max relative drawdown={max_drawdown:.4f} vs limit={limit:.4f}",
    )


@dataclass
class KillSwitch:
    triggered: bool = False
    trigger_reasons: list[str] = field(default_factory=list)

    def check(self, results: list[LimitCheckResult]) -> bool:
        for result in results:
            if result.breached:
                self.triggered = True
                if result.name not in self.trigger_reasons:
                    self.trigger_reasons.append(result.name)
        return self.triggered

    def reset(self) -> None:
        self.triggered = False
        self.trigger_reasons = []
