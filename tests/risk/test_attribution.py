import pandas as pd
import pytest

from risk.attribution import brinson_fachler_attribution, factor_exposure_differential


def test_reconciles_exactly_with_full_sector_and_return_coverage():
    portfolio_weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
    benchmark_weights = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    period_returns = pd.Series({"A": 0.10, "B": -0.05, "C": 0.02, "D": 0.08})
    sector_by_symbol = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"}

    result = brinson_fachler_attribution(portfolio_weights, benchmark_weights, period_returns, sector_by_symbol)
    assert result.reconciles(tol=1e-9)
    assert result.excluded_symbols == []
    assert result.excluded_weight == pytest.approx(0.0)


def test_pure_allocation_case_zero_selection_when_sector_returns_equal_component_returns():
    # both names in each sector have the SAME return -> zero within-sector
    # dispersion, so any active return must come entirely from allocation
    portfolio_weights = pd.Series({"A": 0.6, "B": 0.4})  # overweight Tech (A) vs benchmark
    benchmark_weights = pd.Series({"A": 0.5, "B": 0.5})
    period_returns = pd.Series({"A": 0.10, "B": 0.02})  # Tech outperforms Health
    sector_by_symbol = {"A": "Tech", "B": "Health"}

    result = brinson_fachler_attribution(portfolio_weights, benchmark_weights, period_returns, sector_by_symbol)
    assert result.selection == pytest.approx(0.0, abs=1e-9)
    assert result.interaction == pytest.approx(0.0, abs=1e-9)
    assert result.allocation > 0  # correctly overweighted the sector that outperformed
    assert result.reconciles()


def test_excluded_symbols_reported_and_weights_renormalized():
    portfolio_weights = pd.Series({"A": 0.5, "NOSECTOR": 0.5})
    benchmark_weights = pd.Series({"A": 1.0})
    period_returns = pd.Series({"A": 0.05, "NOSECTOR": 0.20})
    sector_by_symbol = {"A": "Tech"}  # NOSECTOR has no sector -> excluded

    result = brinson_fachler_attribution(portfolio_weights, benchmark_weights, period_returns, sector_by_symbol)
    assert result.excluded_symbols == ["NOSECTOR"]
    assert result.excluded_weight == pytest.approx(0.5)
    assert result.reconciles(tol=1e-9)  # still exact, over the renormalized included subset
    # portfolio_return here is computed over the renormalized (A-only) subset -> equals A's return
    assert result.portfolio_return == pytest.approx(0.05)


def test_by_sector_table_sums_to_scalar_totals():
    portfolio_weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
    benchmark_weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    period_returns = pd.Series({"A": 0.05, "B": -0.02, "C": 0.01})
    sector_by_symbol = {"A": "Tech", "B": "Health", "C": "Health"}

    result = brinson_fachler_attribution(portfolio_weights, benchmark_weights, period_returns, sector_by_symbol)
    assert result.by_sector["allocation"].sum() == pytest.approx(result.allocation)
    assert result.by_sector["selection"].sum() == pytest.approx(result.selection)
    assert result.by_sector["interaction"].sum() == pytest.approx(result.interaction)


def test_factor_exposure_differential_positive_when_portfolio_tilts_toward_high_scoring_names():
    portfolio_weights = pd.Series({"A": 0.8, "B": 0.2})
    benchmark_weights = pd.Series({"A": 0.2, "B": 0.8})
    composite_score = pd.Series({"A": 2.0, "B": -1.0})

    diff = factor_exposure_differential(portfolio_weights, benchmark_weights, composite_score)
    assert diff > 0


def test_factor_exposure_differential_zero_when_weights_identical():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    composite_score = pd.Series({"A": 1.0, "B": -1.0})
    diff = factor_exposure_differential(weights, weights, composite_score)
    assert diff == pytest.approx(0.0)
