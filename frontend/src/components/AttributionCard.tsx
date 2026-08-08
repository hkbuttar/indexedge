import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { AttributionResponse } from '../lib/api'
import { EmptyState } from './Card'

type Props = { data: AttributionResponse | null }

const pct = (v: number) => `${(v * 100).toFixed(2)}%`

export function AttributionCard({ data }: Props) {
  if (!data) return <EmptyState message="No attribution data." />
  const { attribution } = data

  const chartData = [
    { component: 'Allocation', value: attribution.allocation },
    { component: 'Selection', value: attribution.selection },
    { component: 'Interaction', value: attribution.interaction },
  ]

  return (
    <div>
      <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
        {data.strategy} vs. full replication, {new Date(data.period_start).toLocaleDateString()} &rarr;{' '}
        {new Date(data.period_end).toLocaleDateString()}. Active return {pct(attribution.total_active_return)} = allocation +
        selection + interaction (reconciles exactly).
      </p>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
          <CartesianGrid stroke="var(--gridline)" horizontal={false} />
          <XAxis type="number" stroke="var(--baseline)" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} tickFormatter={pct} />
          <YAxis type="category" dataKey="component" stroke="var(--baseline)" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} width={80} />
          <ReferenceLine x={0} stroke="var(--baseline)" />
          <Tooltip
            contentStyle={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 13 }}
            labelStyle={{ color: 'var(--text-secondary)' }}
            formatter={(value) => pct(Number(value))}
          />
          <Legend wrapperStyle={{ fontSize: 13, color: 'var(--text-secondary)' }} />
          <Bar dataKey="value" name="Contribution to active return" fill="var(--series-1)" radius={[3, 3, 3, 3]} barSize={28} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      {attribution.excluded_symbols.length > 0 && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
          {attribution.excluded_symbols.length} symbols excluded (no known sector), {pct(attribution.excluded_weight)} of portfolio weight.
        </p>
      )}
      {data.factor_exposure_differential != null && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Factor exposure differential (portfolio &minus; benchmark): {data.factor_exposure_differential.toFixed(3)}
        </p>
      )}
    </div>
  )
}
