import numpy as np
import pandas as pd
import pytest

from smartbeta.quality import compute_quality_scores, quality_weights


def test_higher_roe_and_lower_leverage_score_higher():
    fundamentals = pd.DataFrame({
        "returnOnEquity": [0.30, 0.05, 0.15],
        "profitMargins": [0.25, 0.05, 0.15],
        "earningsGrowth": [0.20, 0.02, 0.10],
        "debtToEquity": [20.0, 200.0, 80.0],
    }, index=["GOOD", "BAD", "MID"])

    scores = compute_quality_scores(fundamentals)
    assert scores["GOOD"] > scores["MID"] > scores["BAD"]


def test_missing_field_uses_coverage_aware_average():
    fundamentals = pd.DataFrame({
        "returnOnEquity": [0.30, 0.10],
        "profitMargins": [0.25, np.nan],  # missing for the second name
        "earningsGrowth": [0.20, 0.10],
        "debtToEquity": [20.0, 50.0],
    }, index=["A", "B"])
    scores = compute_quality_scores(fundamentals)
    assert scores.notna().all()


def test_quality_weights_are_positive_and_sum_to_one():
    scores = pd.Series({"A": 1.5, "B": -1.0, "C": 0.0})
    weights = quality_weights({"A", "B", "C"}, scores)
    assert (weights > 0).all()  # exp transform: even below-average names get nonzero weight
    assert weights.sum() == pytest.approx(1.0)
    assert weights["A"] > weights["C"] > weights["B"]


def test_quality_weights_empty_when_no_scores_available():
    weights = quality_weights({"X"}, pd.Series(dtype=float))
    assert weights.empty
