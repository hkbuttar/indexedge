import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { RegimePerformanceRow } from '../lib/api'
import { REGIME_ORDER, STRATEGY_COLOR, STRATEGY_LABEL } from '../lib/colors'
import { EmptyState } from './Card'

type Props = { rows: RegimePerformanceRow[] }

const pctFormatter = (v: number) => `${(v * 100).toFixed(1)}%`

export function SmartBetaRegimeChart({ rows }: Props) {
  if (rows.length === 0) return <EmptyState message="No regime-conditional data." />

  const strategies = [...new Set(rows.map((r) => r.strategy))].sort(
    (a, b) => Object.keys(STRATEGY_COLOR).indexOf(a) - Object.keys(STRATEGY_COLOR).indexOf(b),
  )
  const data = REGIME_ORDER.map((regime) => {
    const row: Record<string, string | number> = { regime }
    for (const strategy of strategies) {
      const match = rows.find((r) => r.strategy === strategy && r.regime === regime)
      if (match) row[strategy] = match.annualized_return
    }
    return row
  })

  return (
    <ResponsiveContainer width="100%" height={360}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis dataKey="regime" stroke="var(--baseline)" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
        <YAxis
          stroke="var(--baseline)"
          tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
          tickFormatter={pctFormatter}
          width={56}
          label={{ value: 'Annualized return', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 13 }}
          labelStyle={{ color: 'var(--text-secondary)' }}
          formatter={(value, name) => [pctFormatter(Number(value)), STRATEGY_LABEL[String(name)] ?? name]}
        />
        <Legend wrapperStyle={{ fontSize: 13, color: 'var(--text-secondary)' }} formatter={(value) => STRATEGY_LABEL[value] ?? value} />
        {strategies.map((strategy) => (
          <Bar key={strategy} dataKey={strategy} name={strategy} fill={STRATEGY_COLOR[strategy]} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}
