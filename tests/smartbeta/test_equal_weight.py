import pandas as pd
import pytest

from smartbeta.equal_weight import equal_weights


def test_equal_weights_splits_evenly_across_tradeable_members():
    members = {"A", "B", "C", "D"}
    price_row = pd.Series({"A": 10.0, "B": 20.0, "C": 0.0, "D": float("nan")})
    weights = equal_weights(members, price_row)
    assert set(weights.index) == {"A", "B"}
    assert weights["A"] == pytest.approx(0.5)
    assert weights.sum() == pytest.approx(1.0)


def test_equal_weights_empty_when_nothing_tradeable():
    weights = equal_weights({"X"}, pd.Series({"X": float("nan")}))
    assert weights.empty
