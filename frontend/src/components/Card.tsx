import type { ReactNode } from 'react'

export function Card({ title, subtitle, children }: { title?: string; subtitle?: string; children: ReactNode }) {
  return (
    <div
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        padding: 20,
      }}
    >
      {title && (
        <h3 style={{ fontSize: 14, color: 'var(--text-secondary)', marginBottom: subtitle ? 4 : 16 }}>{title}</h3>
      )}
      {subtitle && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 0, marginBottom: 16 }}>{subtitle}</p>
      )}
      {children}
    </div>
  )
}

export function EmptyState({ message }: { message: string }) {
  return <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>{message}</p>
}

export function LoadingState({ message = 'Loading…' }: { message?: string }) {
  return <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>{message}</p>
}
