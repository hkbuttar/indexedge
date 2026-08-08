export function StatTile({ label, value, tone }: { label: string; value: string; tone?: 'good' | 'critical' }) {
  return (
    <div>
      <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</p>
      <p style={{ fontSize: 28, fontWeight: 600, color: tone ? `var(--status-${tone})` : 'var(--text-primary)' }}>{value}</p>
    </div>
  )
}
