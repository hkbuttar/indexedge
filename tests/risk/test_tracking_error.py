import numpy as np
import pandas as pd

from risk.tracking_error import active_returns, summarize_tracking


def test_tracking_statistics_use_only_complete_overlapping_observations():
    dates = pd.date_range("2024-01-01", periods=4)
    portfolio = pd.Series([0.01, np.nan, 0.03, 0.04], index=dates)
    benchmark = pd.Series([0.00, 0.02, np.nan, 0.01], index=dates)

    active = active_returns(portfolio, benchmark)
    summary = summarize_tracking(portfolio, benchmark)

    assert active.index.tolist() == [dates[0], dates[3]]
    assert summary.n_periods == 2
