import numpy as np
import pandas as pd
import pytest

from regime.conditional import (
    align_regime_labels,
    regime_conditional_stats,
    split_by_regime,
    summarize_regime_conditional_performance,
)


def test_align_regime_labels_does_not_forward_fill():
    returns_index = pd.date_range("2024-01-01", periods=5)
    regime_labels = pd.Series(["calm", "calm"], index=returns_index[:2])
    aligned = align_regime_labels(returns_index, regime_labels)
    assert aligned.iloc[:2].tolist() == ["calm", "calm"]
    assert aligned.iloc[2:].isna().all()  # not forward-filled from the last known label


def test_split_by_regime_groups_correctly():
    dates = pd.date_range("2024-01-01", periods=6)
    returns = pd.Series(range(6), index=dates, dtype=float)
    labels = pd.Series(["calm", "calm", "volatile", "volatile", "normal", np.nan], index=dates)
    groups = split_by_regime(returns, labels)
    assert list(groups["calm"]) == [0.0, 1.0]
    assert list(groups["volatile"]) == [2.0, 3.0]
    assert "nan" not in groups  # unlabeled days excluded entirely


def test_regime_conditional_stats_skips_thin_regimes():
    dates = pd.date_range("2024-01-01", periods=40)
    returns = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 40), index=dates)
    labels = pd.Series(["calm"] * 35 + ["volatile"] * 5, index=dates)  # volatile has only 5 days
    stats, notes = regime_conditional_stats(returns, labels)
    assert "calm" in stats
    assert "volatile" not in stats
    assert any("volatile" in note for note in notes)


def test_regime_conditional_stats_annualization_matches_hand_calc():
    dates = pd.date_range("2024-01-01", periods=35)
    returns = pd.Series(0.001, index=dates)  # constant daily return, exact hand-computable
    labels = pd.Series("calm", index=dates)
    stats, _ = regime_conditional_stats(returns, labels)
    expected_ann_return = (1.001) ** 252 - 1
    assert stats["calm"]["annualized_return"] == pytest.approx(expected_ann_return)
    assert stats["calm"]["annualized_vol"] == pytest.approx(0.0, abs=1e-9)


def test_summarize_regime_conditional_performance_combines_strategies():
    dates = pd.date_range("2024-01-01", periods=80)
    labels = pd.Series(["calm"] * 40 + ["volatile"] * 40, index=dates)
    returns_by_strategy = {
        "strat_a": pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 80), index=dates),
        "strat_b": pd.Series(np.random.default_rng(2).normal(0.0005, 0.02, 80), index=dates),
    }
    summary = summarize_regime_conditional_performance(returns_by_strategy, labels)
    assert set(summary["strategy"].unique()) == {"strat_a", "strat_b"}
    assert set(summary["regime"].unique()) == {"calm", "volatile"}
    assert len(summary) == 4
