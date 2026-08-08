import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import type { FrontierRow, MultiObjectiveResponse } from '../lib/api'
import { FACTOR_TARGET_COLOR, FACTOR_TARGET_LABEL } from '../lib/colors'
import { EmptyState } from './Card'

type Props = { data: MultiObjectiveResponse }

const pctFormatter = (v: number) => `${(v * 100).toFixed(2)}%`

function seriesFor(frontier: FrontierRow[], target: number) {
  return frontier
    .filter((r) => r.factor_target === target)
    .sort((a, b) => a.turnover_budget - b.turnover_budget)
}

export function ParetoFrontierChart({ data }: Props) {
  if (data.frontier.length === 0) return <EmptyState message="No frontier data." />

  const groups: { key: 'unconstrained' | 'median' | 'p75'; target: number }[] = [
    { key: 'unconstrained', target: data.factor_targets.unconstrained },
    { key: 'median', target: data.factor_targets.median },
    { key: 'p75', target: data.factor_targets.p75 },
  ]

  return (
    <ResponsiveContainer width="100%" height={400}>
    <ScatterChart margin={{ top: 8, right: 24, left: 8, bottom: 36 }}>
      <CartesianGrid stroke="var(--gridline)" />
      <XAxis
        type="number"
        dataKey="realized_turnover"
        stroke="var(--baseline)"
        tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
        tickFormatter={pctFormatter}
        label={{ value: 'Realized turnover', position: 'bottom', offset: 12, fill: 'var(--text-muted)', fontSize: 12 }}
      />
      <YAxis
        type="number"
        dataKey="tracking_error"
        stroke="var(--baseline)"
        tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
        tickFormatter={pctFormatter}
        width={64}
        label={{ value: 'Tracking error', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 12 }}
      />
      <ZAxis range={[64, 64]} />
      <Tooltip
        cursor={{ stroke: 'var(--baseline)' }}
        contentStyle={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 13 }}
        labelStyle={{ color: 'var(--text-secondary)' }}
        formatter={(value, name) => [pctFormatter(Number(value)), name]}
      />
      <Legend verticalAlign="top" height={32} wrapperStyle={{ fontSize: 13, color: 'var(--text-secondary)' }} />
      {groups.map(({ key, target }) => (
        <Scatter
          key={key}
          name={FACTOR_TARGET_LABEL[key]}
          data={seriesFor(data.frontier, target)}
          fill={FACTOR_TARGET_COLOR[key]}
          line={{ stroke: FACTOR_TARGET_COLOR[key], strokeWidth: 2 }}
          shape="circle"
          isAnimationActive={false}
        />
      ))}
    </ScatterChart>
    </ResponsiveContainer>
  )
}
