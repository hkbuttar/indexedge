import { Bar, BarChart, CartesianGrid, ErrorBar, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ComparisonRow } from '../lib/api'
import { STRATEGY_COLOR, STRATEGY_LABEL } from '../lib/colors'
import { EmptyState } from './Card'

type Props = { rows: ComparisonRow[] }

const pctFormatter = (v: number) => `${(v * 100).toFixed(1)}%`
const aumLabel = (aum: number) => (aum >= 1e9 ? `$${aum / 1e9}B` : `$${aum / 1e6}M`)

export function CapacityChart({ rows }: Props) {
  const allRegime = rows.filter((r) => r.regime === 'all')
  if (allRegime.length === 0) return <EmptyState message="No capacity comparison data." />

  const aums = [...new Set(allRegime.map((r) => r.aum))].sort((a, b) => a - b)
  const strategies = [...new Set(allRegime.map((r) => r.strategy))].sort(
    (a, b) => Object.keys(STRATEGY_COLOR).indexOf(a) - Object.keys(STRATEGY_COLOR).indexOf(b),
  )

  return (
    <div style={{ display: 'grid', gap: 24 }}>
      {strategies.map((strategy) => {
        const data = aums.map((aum) => {
          const row = allRegime.find((r) => r.strategy === strategy && r.aum === aum)
          if (!row) return { aum: aumLabel(aum) }
          return {
            aum: aumLabel(aum),
            cost_adjusted_return: row.cost_adjusted_return,
            errorRange: [row.cost_adjusted_return - row.cost_adjusted_return_ci_low, row.cost_adjusted_return_ci_high - row.cost_adjusted_return],
          }
        })
        return (
          <div key={strategy}>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 4 }}>{STRATEGY_LABEL[strategy] ?? strategy}</p>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
                <CartesianGrid stroke="var(--gridline)" horizontal={false} />
                <XAxis
                  type="number"
                  stroke="var(--baseline)"
                  tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                  tickFormatter={pctFormatter}
                  domain={['auto', 'auto']}
                />
                <YAxis type="category" dataKey="aum" stroke="var(--baseline)" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} width={48} />
                <ReferenceLine x={0} stroke="var(--baseline)" />
                <Tooltip
                  contentStyle={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 13 }}
                  labelStyle={{ color: 'var(--text-secondary)' }}
                  formatter={(value) => [pctFormatter(Number(value)), 'Cost-adjusted CAGR']}
                />
                <Bar dataKey="cost_adjusted_return" fill={STRATEGY_COLOR[strategy]} radius={[3, 3, 3, 3]} barSize={18} isAnimationActive={false}>
                  <ErrorBar dataKey="errorRange" stroke="var(--text-muted)" width={4} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )
      })}
    </div>
  )
}
