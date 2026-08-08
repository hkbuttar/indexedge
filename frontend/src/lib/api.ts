const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export type TrackingSummary = {
  tracking_error_annualized: number
  mean_active_return_annualized: number
  correlation: number
  n_periods: number
  cumulative_active_return: number
}

export type ReplicationFullResponse = {
  vs_price_index: TrackingSummary
  vs_total_return_index: TrackingSummary
  coverage_by_rebalance: { rebalance_date: string; intended_members: number; weighted_members: number; coverage_fraction: number }[]
}

export type SamplingCurveRow = {
  method: 'stratified' | 'optimization' | 'lasso'
  target_n: number
  mean_actual_n: number
  mean_tracking_error: number
  median_tracking_error: number
  n_periods: number
}
export type SamplingResponse = { target_counts: number[]; curve: SamplingCurveRow[] }

export type FrontierRow = {
  factor_target: number
  turnover_budget: number
  realized_turnover: number
  realized_factor_exposure: number
  tracking_error: number
  n_holdings: number
}
export type MultiObjectiveResponse = {
  rebalance_date: string
  n_candidates: number
  factor_targets: { unconstrained: number; median: number; p75: number }
  frontier: FrontierRow[]
}

export type SmartBetaRow = {
  strategy: string
  annualized_return: number
  annualized_vol: number
  tracking_error_vs_full_replication: number
  correlation: number
}
export type SmartBetaResponse = { strategies: SmartBetaRow[] }

export type RegimePerformanceRow = {
  strategy: string
  regime: 'calm' | 'normal' | 'volatile'
  annualized_return: number
  annualized_vol: number
  sharpe_zero_rate: number
  n_days: number
}
export type RegimeResponse = {
  regime_day_counts: { calm: number; normal: number; volatile: number }
  current_regime: string | null
  performance_by_strategy_and_regime: RegimePerformanceRow[]
}

export type LiquidityRow = {
  strategy: string
  aum: number
  rebalance_date: string
  n_holdings: number
  turnover: number
  rebalance_cost_fraction: number
  annualized_cost_drag: number
}
export type LiquidityResponse = { aum: number; rebalance_date: string; strategies: LiquidityRow[] }

export type BootstrapResult = { point_estimate: number; ci_low: number; ci_high: number; n_resamples: number; confidence_level: number }
export type BootstrapResponse = {
  strategy: string
  aum: number
  n_resamples: number
  mean_rebalance_cost_fraction: number
  metrics: Record<'cagr' | 'sharpe_ratio' | 'sortino_ratio' | 'max_drawdown' | 'win_rate' | 'tracking_error', BootstrapResult>
}

export type LimitCheckResult = { name: string; breached: boolean; detail: string }
export type KillSwitchRow = {
  strategy: string
  tracking_error_check: LimitCheckResult
  relative_drawdown_check: LimitCheckResult
  triggered: boolean
  trigger_reasons: string[]
}
export type KillSwitchResponse = { aum: number; strategies: KillSwitchRow[] }

export type BrinsonAttribution = {
  allocation: number
  selection: number
  interaction: number
  total_active_return: number
  portfolio_return: number
  benchmark_return: number
  excluded_symbols: string[]
  excluded_weight: number
}
export type AttributionResponse = {
  strategy: string
  period_start: string
  period_end: string
  attribution: BrinsonAttribution
  factor_exposure_differential: number | null
}

export type ComparisonRow = {
  strategy: string
  strategy_family: string
  aum: number
  regime: string
  name_count: number | null
  mean_one_way_turnover: number | null
  n_days: number
  gross_annualized_return: number
  cost_adjusted_return: number
  cost_adjusted_return_ci_low: number
  cost_adjusted_return_ci_high: number
  sharpe: number
  sharpe_ci_low: number
  sharpe_ci_high: number
  tracking_error: number
  tracking_error_ci_low: number
  tracking_error_ci_high: number
}
export type ResultsResponse = {
  comparison: ComparisonRow[]
  sampling_comparison: SamplingCurveRow[]
  findings: string[]
  error?: string
}

async function getJSON<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`)
  if (params) {
    for (const [key, value] of Object.entries(params)) url.searchParams.set(key, String(value))
  }
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`${path} failed with status ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  replicationFull: () => getJSON<ReplicationFullResponse>('/api/replication/full'),
  sampling: (targetCounts: number[]) => getJSON<SamplingResponse>('/api/replication/sampling', { target_counts: targetCounts.join(',') }),
  multiObjective: (turnoverBudgets: number[]) => getJSON<MultiObjectiveResponse>('/api/multi-objective', { turnover_budgets: turnoverBudgets.join(',') }),
  smartbeta: () => getJSON<SmartBetaResponse>('/api/smartbeta'),
  regime: () => getJSON<RegimeResponse>('/api/regime'),
  liquidity: (aum: number) => getJSON<LiquidityResponse>('/api/liquidity', { aum }),
  bootstrap: (strategy: string, aum: number, nResamples = 500) =>
    getJSON<BootstrapResponse>('/api/backtest/bootstrap', { strategy, aum, n_resamples: nResamples }),
  attribution: (strategy = 'multi_factor') => getJSON<AttributionResponse>('/api/risk/attribution', { strategy }),
  killSwitch: (aum: number) => getJSON<KillSwitchResponse>('/api/risk/kill-switch', { aum }),
  results: () => getJSON<ResultsResponse>('/api/results'),
}
