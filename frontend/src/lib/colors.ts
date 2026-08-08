// Fixed entity -> categorical slot assignments, used identically across
// every chart in the app (dataviz skill rule: color follows the entity,
// never its rank -- a strategy keeps its color whether it's ranked 1st or
// last, and whether or not other strategies are present in a given view).

export const STRATEGY_COLOR: Record<string, string> = {
  full_replication: 'var(--series-1)',
  equal_weight: 'var(--series-2)',
  min_vol: 'var(--series-3)',
  quality: 'var(--series-4)',
  multi_factor: 'var(--series-5)',
}

export const STRATEGY_LABEL: Record<string, string> = {
  full_replication: 'Full replication',
  equal_weight: 'Equal-weight',
  min_vol: 'Min-vol',
  quality: 'Quality-weighted',
  multi_factor: 'Multi-factor tilt',
}

export const SAMPLING_METHOD_COLOR: Record<string, string> = {
  stratified: 'var(--series-1)',
  optimization: 'var(--series-2)',
  lasso: 'var(--series-3)',
}

export const SAMPLING_METHOD_LABEL: Record<string, string> = {
  stratified: 'Stratified',
  optimization: 'Optimization (cvxpy)',
  lasso: 'LASSO',
}

// Pareto frontier scatter uses only the first three slots -- validated
// all-pairs (see dataviz skill palette.md: scatter/bubble forms cap at 3).
export const FACTOR_TARGET_COLOR: Record<'unconstrained' | 'median' | 'p75', string> = {
  unconstrained: 'var(--series-1)',
  median: 'var(--series-2)',
  p75: 'var(--series-3)',
}

export const FACTOR_TARGET_LABEL: Record<'unconstrained' | 'median' | 'p75', string> = {
  unconstrained: 'Unconstrained',
  median: 'Median factor exposure',
  p75: '75th pctile factor exposure',
}

export const REGIME_ORDER = ['calm', 'normal', 'volatile'] as const
