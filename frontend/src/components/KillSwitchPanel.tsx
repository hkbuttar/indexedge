import type { KillSwitchRow } from '../lib/api'
import { STRATEGY_LABEL } from '../lib/colors'
import { EmptyState } from './Card'

type Props = { rows: KillSwitchRow[] }

export function KillSwitchPanel({ rows }: Props) {
  if (rows.length === 0) return <EmptyState message="No kill-switch data." />

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {rows.map((row) => (
        <div
          key={row.strategy}
          style={{
            display: 'flex', alignItems: 'flex-start', gap: 12, padding: '10px 12px',
            border: '1px solid var(--border)', borderRadius: 6,
          }}
        >
          <span
            aria-hidden
            style={{
              width: 10, height: 10, borderRadius: '50%', marginTop: 4, flexShrink: 0,
              background: row.triggered ? 'var(--status-critical)' : 'var(--status-good)',
            }}
          />
          <div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>
              {STRATEGY_LABEL[row.strategy] ?? row.strategy} —{' '}
              <span style={{ color: row.triggered ? 'var(--status-critical)' : 'var(--status-good)' }}>
                {row.triggered ? 'Triggered' : 'OK'}
              </span>
              {row.triggered && (
                <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> ({row.trigger_reasons.join(', ')})</span>
              )}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>{row.tracking_error_check.detail}</div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{row.relative_drawdown_check.detail}</div>
          </div>
        </div>
      ))}
    </div>
  )
}
