import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { SamplingCurveRow } from '../lib/api'
import { SAMPLING_METHOD_COLOR, SAMPLING_METHOD_LABEL } from '../lib/colors'
import { EmptyState } from './Card'

type Props = { curve: SamplingCurveRow[] }

const METHODS = ['stratified', 'optimization', 'lasso'] as const

const pctFormatter = (v: number) => `${(v * 100).toFixed(1)}%`

export function SamplingCurveChart({ curve }: Props) {
  if (curve.length === 0) return <EmptyState message="No sampling comparison data." />

  const targetNs = [...new Set(curve.map((r) => r.target_n))].sort((a, b) => a - b)
  const data = targetNs.map((n) => {
    const row: Record<string, number> = { target_n: n }
    for (const method of METHODS) {
      const match = curve.find((r) => r.method === method && r.target_n === n)
      if (match) row[method] = match.mean_tracking_error
    }
    return row
  })

  return (
    <ResponsiveContainer width="100%" height={340}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis
          dataKey="target_n"
          stroke="var(--baseline)"
          tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
          label={{ value: 'Target name count', position: 'insideBottom', offset: -4, fill: 'var(--text-muted)', fontSize: 12 }}
        />
        <YAxis
          stroke="var(--baseline)"
          tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
          tickFormatter={pctFormatter}
          width={56}
          label={{ value: 'Mean tracking error', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 13 }}
          labelStyle={{ color: 'var(--text-secondary)' }}
          labelFormatter={(v) => `${v} names`}
          formatter={(value) => pctFormatter(Number(value))}
        />
        <Legend wrapperStyle={{ fontSize: 13, color: 'var(--text-secondary)' }} formatter={(value) => SAMPLING_METHOD_LABEL[value] ?? value} />
        {METHODS.map((method) => (
          <Line
            key={method}
            type="monotone"
            dataKey={method}
            name={method}
            stroke={SAMPLING_METHOD_COLOR[method]}
            strokeWidth={2}
            dot={{ r: 4, strokeWidth: 0, fill: SAMPLING_METHOD_COLOR[method] }}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
