"""Contract tests for the FastAPI layer, against the real app instance and
real locally-cached data (`data/cache/`) -- no network calls happen inside
any route handler (even `/api/replication/full`'s ^GSPC/^SP500TR lookups
read from the same local index cache the data pipeline populated), so these are
naturally network-independent without needing to monkeypatch anything, the
way riskdesk's equivalent tests must (riskdesk's routes fetch live).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as api

client = TestClient(api.app)

REQUIRED_ROUTES = {
    "/health",
    "/api/replication/full",
    "/api/replication/sampling",
    "/api/multi-objective",
    "/api/smartbeta",
    "/api/regime",
    "/api/liquidity",
    "/api/backtest/bootstrap",
    "/api/risk/attribution",
    "/api/risk/kill-switch",
    "/api/results",
}


def test_openapi_exposes_every_route():
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert REQUIRED_ROUTES <= set(schema.json()["paths"])
    assert schema.json()["info"]["version"] == "1.0.0"


def test_health_is_a_stable_machine_readable_probe():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "indexedge", "version": "1.0.0"}


def test_replication_full_reports_both_benchmarks():
    response = client.get("/api/replication/full")
    assert response.status_code == 200
    body = response.json()
    assert "vs_price_index" in body and "vs_total_return_index" in body
    assert body["vs_price_index"]["tracking_error_annualized"] > 0
    assert len(body["coverage_by_rebalance"]) > 0


def test_replication_sampling_respects_target_counts_param():
    response = client.get("/api/replication/sampling", params={"target_counts": "30,60"})
    assert response.status_code == 200
    body = response.json()
    assert body["target_counts"] == [30, 60]
    methods = {row["method"] for row in body["curve"]}
    assert methods == {"stratified", "optimization", "lasso"}


def test_multi_objective_frontier_is_monotonic_in_turnover_budget():
    response = client.get("/api/multi-objective", params={"turnover_budgets": "0.1,0.5,1.0"})
    assert response.status_code == 200
    frontier = response.json()["frontier"]
    unconstrained = sorted(
        (row for row in frontier if row["factor_target"] == response.json()["factor_targets"]["unconstrained"]),
        key=lambda r: r["turnover_budget"],
    )
    tes = [row["tracking_error"] for row in unconstrained]
    for earlier, later in zip(tes, tes[1:]):
        assert later <= earlier + 1e-6


def test_smartbeta_reports_all_four_variants():
    response = client.get("/api/smartbeta")
    assert response.status_code == 200
    strategies = {row["strategy"] for row in response.json()["strategies"]}
    assert strategies == {"equal_weight", "min_vol", "quality", "multi_factor"}
    for row in response.json()["strategies"]:
        assert row["annualized_vol"] > 0


def test_regime_reports_calm_normal_volatile():
    response = client.get("/api/regime")
    assert response.status_code == 200
    body = response.json()
    assert set(body["regime_day_counts"].keys()) == {"calm", "normal", "volatile"}
    regimes_seen = {row["regime"] for row in body["performance_by_strategy_and_regime"]}
    assert regimes_seen <= {"calm", "normal", "volatile"}


def test_liquidity_cost_scales_with_aum():
    small = client.get("/api/liquidity", params={"aum": 10_000_000}).json()
    large = client.get("/api/liquidity", params={"aum": 1_000_000_000}).json()
    small_by_strategy = {row["strategy"]: row["annualized_cost_drag"] for row in small["strategies"]}
    large_by_strategy = {row["strategy"]: row["annualized_cost_drag"] for row in large["strategies"]}
    for strategy in small_by_strategy:
        assert large_by_strategy[strategy] >= small_by_strategy[strategy]


def test_liquidity_rejects_nonpositive_aum():
    response = client.get("/api/liquidity", params={"aum": 0})
    assert response.status_code == 422


def test_backtest_bootstrap_ci_brackets_point_estimate():
    response = client.get("/api/backtest/bootstrap", params={"strategy": "multi_factor", "n_resamples": 200})
    assert response.status_code == 200
    cagr = response.json()["metrics"]["cagr"]
    assert cagr["ci_low"] <= cagr["point_estimate"] <= cagr["ci_high"]


def test_backtest_bootstrap_rejects_unknown_strategy():
    response = client.get("/api/backtest/bootstrap", params={"strategy": "not_a_real_strategy"})
    assert response.status_code == 200  # returns a JSON error body, not an HTTP error, by design
    assert "error" in response.json()


def test_risk_attribution_reconciles():
    response = client.get("/api/risk/attribution", params={"strategy": "multi_factor"})
    assert response.status_code == 200
    body = response.json()["attribution"]
    total = body["allocation"] + body["selection"] + body["interaction"]
    assert abs(total - body["total_active_return"]) < 1e-6


def test_risk_kill_switch_reports_all_strategies():
    response = client.get("/api/risk/kill-switch", params={"aum": 100_000_000})
    assert response.status_code == 200
    strategies = {row["strategy"] for row in response.json()["strategies"]}
    assert strategies == {"equal_weight", "min_vol", "quality", "multi_factor"}


def test_results_serves_precomputed_output_when_present():
    response = client.get("/api/results")
    assert response.status_code == 200
    body = response.json()
    assert "error" not in body
    assert len(body["comparison"]) > 0
    assert len(body["findings"]) > 0
