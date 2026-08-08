import { useMemo, useState } from 'react'
import type { ComparisonRow } from '../lib/api'
import { STRATEGY_LABEL } from '../lib/colors'
import { EmptyState } from './Card'

type Props = { rows: ComparisonRow[] }

const pct = (v: number) => `${(v * 100).toFixed(2)}%`
const aumLabel = (aum: number) => (aum >= 1e9 ? `$${aum / 1e9}B` : `$${aum / 1e6}M`)

export function ResultsTable({ rows }: Props) {
  const [regime, setRegime] = useState('all')
  const [aum, setAum] = useState<number>(rows[0]?.aum ?? 1e8)

  const aums = useMemo(() => [...new Set(rows.map((r) => r.aum))].sort((a, b) => a - b), [rows])
  const regimes = useMemo(() => [...new Set(rows.map((r) => r.regime))], [rows])

  const filtered = rows.filter((r) => r.regime === regime && r.aum === aum)
  if (rows.length === 0) return <EmptyState message="No results table available -- run `python -m results.run_full_comparison` first." />

  return (
    <div>
      <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
        <label style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
          AUM:{' '}
          <select value={aum} onChange={(e) => setAum(Number(e.target.value))}>
            {aums.map((a) => (
              <option key={a} value={a}>
                {aumLabel(a)}
              </option>
            ))}
          </select>
        </label>
        <label style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
          Regime:{' '}
          <select value={regime} onChange={(e) => setRegime(e.target.value)}>
            {regimes.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
      </div>

      <table>
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Names</th>
            <th>Turnover</th>
            <th>Cost-adj. CAGR (95% CI)</th>
            <th>Sharpe (95% CI)</th>
            <th>Tracking error (95% CI)</th>
          </tr>
        </thead>
        <tbody>
          {filtered
            .sort((a, b) => b.cost_adjusted_return - a.cost_adjusted_return)
            .map((row) => (
              <tr key={row.strategy}>
                <td>{STRATEGY_LABEL[row.strategy] ?? row.strategy}</td>
                <td>{row.name_count ? Math.round(row.name_count) : '—'}</td>
                <td>{row.mean_one_way_turnover != null ? pct(row.mean_one_way_turnover) : '—'}</td>
                <td>
                  {pct(row.cost_adjusted_return)}{' '}
                  <span style={{ color: 'var(--text-muted)' }}>
                    [{pct(row.cost_adjusted_return_ci_low)}, {pct(row.cost_adjusted_return_ci_high)}]
                  </span>
                </td>
                <td>
                  {row.sharpe.toFixed(2)} <span style={{ color: 'var(--text-muted)' }}>[{row.sharpe_ci_low.toFixed(2)}, {row.sharpe_ci_high.toFixed(2)}]</span>
                </td>
                <td>
                  {pct(row.tracking_error)}{' '}
                  <span style={{ color: 'var(--text-muted)' }}>
                    [{pct(row.tracking_error_ci_low)}, {pct(row.tracking_error_ci_high)}]
                  </span>
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  )
}
