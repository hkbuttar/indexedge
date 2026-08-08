"""Active-risk decomposition: splits a portfolio's active return (vs a
benchmark, over one holding period) into sector allocation, security
selection, and interaction effects -- classic Brinson-Fachler attribution.

No sibling project implements this (riskdesk's `attribution/` module is a
factor-regression decomposition of dollar P&L for a discretionary
multi-strategy book, a different problem -- see that module for why it
doesn't transfer directly: it explains P&L by named risk factor, not active
return by sector tilt vs. name selection). This is new methodology for the
portfolio, built to decompose active return into "sector tilt vs.
name selection," which Brinson-Fachler is the standard answer to.

    Allocation_s = (w_p,s - w_b,s) * (R_b,s - R_b)   # sector over/underweight, times how that sector did vs the total benchmark
    Selection_s  = w_b,s * (R_p,s - R_b,s)            # picking better/worse names within a sector, at the benchmark's own sector weight
    Interaction_s = (w_p,s - w_b,s) * (R_p,s - R_b,s) # cross term: both over/underweighting AND picking differently in the same sector

Allocation + Selection + Interaction reconciles EXACTLY to the total active
return (R_p - R_b), by algebraic identity given both weight sets sum to 1 --
not a regression residual "plugged" to force reconciliation the way
riskdesk's factor attribution is, a genuinely tighter guarantee. Proven
directly in `tests/risk/test_attribution.py`.

Sector labels are the same current-snapshot GICS sector limitation
disclosed in `replication/stratified.py` and `data/fundamentals.py`:
symbols with no known sector (or no known period return) are excluded from
the decomposition, not assigned a fabricated "unknown" bucket. Because the
exact-reconciliation identity requires each side's included weights to sum
to 1, weights are renormalized over the included (sector-and-return-known)
subset -- so this attributes the covered portion of the portfolio, with the
excluded weight reported via `excluded_symbols`/`excluded_weight` rather
than silently folded in or dropped unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BrinsonAttribution:
    allocation: float
    selection: float
    interaction: float
    total_active_return: float
    portfolio_return: float
    benchmark_return: float
    by_sector: pd.DataFrame
    excluded_symbols: list[str] = field(default_factory=list)
    excluded_weight: float = 0.0

    def reconciles(self, tol: float = 1e-9) -> bool:
        return abs((self.allocation + self.selection + self.interaction) - self.total_active_return) < tol


def brinson_fachler_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    period_returns: pd.Series,
    sector_by_symbol: dict[str, str],
) -> BrinsonAttribution:
    all_symbols = set(portfolio_weights.index) | set(benchmark_weights.index)
    known_sector = {s for s in all_symbols if s in sector_by_symbol}
    has_return = {s for s in known_sector if s in period_returns.index and pd.notna(period_returns[s])}
    excluded = sorted(all_symbols - has_return)
    excluded_weight = float(sum(portfolio_weights.get(s, 0.0) for s in excluded))

    portfolio_weight_raw = pd.Series({s: portfolio_weights.get(s, 0.0) for s in has_return})
    benchmark_weight_raw = pd.Series({s: benchmark_weights.get(s, 0.0) for s in has_return})
    portfolio_total = portfolio_weight_raw.sum()
    benchmark_total = benchmark_weight_raw.sum()

    df = pd.DataFrame({
        "portfolio_weight": portfolio_weight_raw / portfolio_total if portfolio_total > 0 else portfolio_weight_raw,
        "benchmark_weight": benchmark_weight_raw / benchmark_total if benchmark_total > 0 else benchmark_weight_raw,
        "return": pd.Series({s: period_returns[s] for s in has_return}),
        "sector": pd.Series({s: sector_by_symbol[s] for s in has_return}),
    })

    def _weighted_sector_return(weight_col: str) -> pd.Series:
        weighted = df.assign(_wr=df[weight_col] * df["return"]).groupby("sector")
        sector_weight = weighted[weight_col].sum()
        sector_return = weighted["_wr"].sum() / sector_weight.replace(0, pd.NA)
        return sector_return.fillna(0.0)

    portfolio_sector_weight = df.groupby("sector")["portfolio_weight"].sum()
    benchmark_sector_weight = df.groupby("sector")["benchmark_weight"].sum()
    portfolio_sector_return = _weighted_sector_return("portfolio_weight")
    benchmark_sector_return = _weighted_sector_return("benchmark_weight")

    sectors = sorted(set(portfolio_sector_weight.index) | set(benchmark_sector_weight.index))
    portfolio_sector_weight = portfolio_sector_weight.reindex(sectors, fill_value=0.0)
    benchmark_sector_weight = benchmark_sector_weight.reindex(sectors, fill_value=0.0)
    portfolio_sector_return = portfolio_sector_return.reindex(sectors, fill_value=0.0)
    benchmark_sector_return = benchmark_sector_return.reindex(sectors, fill_value=0.0)

    portfolio_return = float((df["portfolio_weight"] * df["return"]).sum())
    benchmark_return = float((df["benchmark_weight"] * df["return"]).sum())

    weight_diff = portfolio_sector_weight - benchmark_sector_weight
    return_diff = portfolio_sector_return - benchmark_sector_return

    allocation_by_sector = weight_diff * (benchmark_sector_return - benchmark_return)
    selection_by_sector = benchmark_sector_weight * return_diff
    interaction_by_sector = weight_diff * return_diff

    by_sector = pd.DataFrame({
        "portfolio_weight": portfolio_sector_weight, "benchmark_weight": benchmark_sector_weight,
        "portfolio_return": portfolio_sector_return, "benchmark_return": benchmark_sector_return,
        "allocation": allocation_by_sector, "selection": selection_by_sector, "interaction": interaction_by_sector,
    })

    return BrinsonAttribution(
        allocation=float(allocation_by_sector.sum()),
        selection=float(selection_by_sector.sum()),
        interaction=float(interaction_by_sector.sum()),
        total_active_return=portfolio_return - benchmark_return,
        portfolio_return=portfolio_return, benchmark_return=benchmark_return,
        by_sector=by_sector, excluded_symbols=excluded, excluded_weight=excluded_weight,
    )


def factor_exposure_differential(
    portfolio_weights: pd.Series, benchmark_weights: pd.Series, composite_score: pd.Series
) -> float:
    """Portfolio's weighted-average multi-factor composite score exposure
    minus the benchmark's -- a risk-exposure diagnostic (how tilted is this
    portfolio on the multi-factor composite score, relative to the benchmark),
    not a return decomposition like the Brinson breakdown above."""
    common = composite_score.dropna().index
    p = portfolio_weights.reindex(common).fillna(0.0)
    b = benchmark_weights.reindex(common).fillna(0.0)
    p_exposure = float((p * composite_score[common]).sum() / p.sum()) if p.sum() > 0 else float("nan")
    b_exposure = float((b * composite_score[common]).sum() / b.sum()) if b.sum() > 0 else float("nan")
    return p_exposure - b_exposure
