import { useEffect, useState } from 'react'
import './App.css'
import { AttributionCard } from './components/AttributionCard'
import { Card, LoadingState } from './components/Card'
import { CapacityChart } from './components/CapacityChart'
import { KillSwitchPanel } from './components/KillSwitchPanel'
import { ParetoFrontierChart } from './components/ParetoFrontierChart'
import { ResultsTable } from './components/ResultsTable'
import { SamplingCurveChart } from './components/SamplingCurveChart'
import { SmartBetaRegimeChart } from './components/SmartBetaRegimeChart'
import { StatTile } from './components/StatTile'
import {
  api,
  type AttributionResponse,
  type KillSwitchResponse,
  type MultiObjectiveResponse,
  type RegimeResponse,
  type ReplicationFullResponse,
  type ResultsResponse,
  type SamplingResponse,
  type SmartBetaResponse,
} from './lib/api'
import { STRATEGY_LABEL } from './lib/colors'

type Tab = 'replication' | 'sampling' | 'multi-objective' | 'smartbeta' | 'capacity' | 'results'

const TABS: { id: Tab; label: string }[] = [
  { id: 'replication', label: 'Replication' },
  { id: 'sampling', label: 'Sampling' },
  { id: 'multi-objective', label: 'Multi-Objective' },
  { id: 'smartbeta', label: 'Smart-Beta & Regime' },
  { id: 'capacity', label: 'Capacity & Risk' },
  { id: 'results', label: 'Full Results' },
]

const DEFAULT_AUM = 100_000_000
const pct = (v: number) => `${(v * 100).toFixed(2)}%`

export default function App() {
  const [tab, setTab] = useState<Tab>('replication')
  const [error, setError] = useState<string | null>(null)

  const [replication, setReplication] = useState<ReplicationFullResponse | null>(null)
  const [sampling, setSampling] = useState<SamplingResponse | null>(null)
  const [multiObjective, setMultiObjective] = useState<MultiObjectiveResponse | null>(null)
  const [smartbeta, setSmartbeta] = useState<SmartBetaResponse | null>(null)
  const [regime, setRegime] = useState<RegimeResponse | null>(null)
  const [killSwitch, setKillSwitch] = useState<KillSwitchResponse | null>(null)
  const [attribution, setAttribution] = useState<AttributionResponse | null>(null)
  const [results, setResults] = useState<ResultsResponse | null>(null)

  const fail = (e: unknown) => setError(String(e))

  useEffect(() => {
    if (tab === 'replication' && !replication) api.replicationFull().then(setReplication).catch(fail)
    if (tab === 'sampling' && !sampling) api.sampling([20, 40, 60, 100, 150, 200]).then(setSampling).catch(fail)
    if (tab === 'multi-objective' && !multiObjective)
      api.multiObjective([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]).then(setMultiObjective).catch(fail)
    if (tab === 'smartbeta') {
      if (!smartbeta) api.smartbeta().then(setSmartbeta).catch(fail)
      if (!regime) api.regime().then(setRegime).catch(fail)
    }
    if (tab === 'capacity') {
      if (!killSwitch) api.killSwitch(DEFAULT_AUM).then(setKillSwitch).catch(fail)
      if (!attribution) api.attribution('multi_factor').then(setAttribution).catch(fail)
      if (!results) api.results().then(setResults).catch(fail)
    }
    if (tab === 'results' && !results) api.results().then(setResults).catch(fail)
  }, [tab, replication, sampling, multiObjective, smartbeta, regime, killSwitch, attribution, results])

  return (
    <div className="app">
      <header className="app-header">
        <h1 style={{ fontSize: 22 }}>IndexEdge</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
          S&amp;P 500 replication and smart-beta construction: point-in-time reconstruction, optimized sampling,
          regime-conditional and capacity-aware, bootstrap-validated results.
        </p>
      </header>

      {error && (
        <div
          style={{
            background: 'var(--status-critical)', color: 'white', padding: '10px 16px',
            borderRadius: 6, marginBottom: 16, fontSize: 13,
          }}
        >
          Couldn't reach the API ({error}). Is the backend running (`uvicorn backend.main:app`)?
        </div>
      )}

      <nav className="tab-nav">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? 'tab active' : 'tab'} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === 'replication' &&
          (replication ? (
            <div style={{ display: 'grid', gap: 20 }}>
              <Card title="Full replication vs. real S&P 500">
                <div className="stat-grid">
                  <StatTile label="TE vs. price index (^GSPC)" value={pct(replication.vs_price_index.tracking_error_annualized)} />
                  <StatTile
                    label="TE vs. total-return index (^SP500TR)"
                    value={pct(replication.vs_total_return_index.tracking_error_annualized)}
                  />
                  <StatTile label="Correlation" value={replication.vs_price_index.correlation.toFixed(4)} />
                </div>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 16 }}>
                  Not literally zero -- driven by disclosed data-coverage and free-float-proxy gaps, not a simulation
                  bug (see <code>replication/full_replication.py</code>).
                </p>
              </Card>
              <Card title="Point-in-time membership coverage by rebalance">
                <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Fraction of intended point-in-time constituents with usable market-cap data at each real quarterly
                  rebalance ({replication.coverage_by_rebalance.length} rebalances).
                </p>
              </Card>
            </div>
          ) : (
            <LoadingState />
          ))}

        {tab === 'sampling' && (
          <Card
            title="Tracking error vs. name count"
            subtitle="Walk-forward: fit on trailing data, evaluated out-of-sample, across 37 real rebalance dates."
          >
            {sampling ? <SamplingCurveChart curve={sampling.curve} /> : <LoadingState />}
          </Card>
        )}

        {tab === 'multi-objective' && (
          <Card
            title="Tracking-error-vs-turnover Pareto frontier"
            subtitle="Multi-factor tilt, most recent real rebalance, at three factor-exposure targets."
          >
            {multiObjective ? <ParetoFrontierChart data={multiObjective} /> : <LoadingState />}
          </Card>
        )}

        {tab === 'smartbeta' && (
          <div style={{ display: 'grid', gap: 20 }}>
            <Card title="Smart-beta variants, full backtest">
              {smartbeta ? (
                <table>
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      <th>Ann. return</th>
                      <th>Ann. vol</th>
                      <th>TE vs. full replication</th>
                    </tr>
                  </thead>
                  <tbody>
                    {smartbeta.strategies.map((row) => (
                      <tr key={row.strategy}>
                        <td>{STRATEGY_LABEL[row.strategy] ?? row.strategy}</td>
                        <td>{pct(row.annualized_return)}</td>
                        <td>{pct(row.annualized_vol)}</td>
                        <td>{pct(row.tracking_error_vs_full_replication)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <LoadingState />
              )}
            </Card>
            <Card
              title="Regime-conditional annualized return"
              subtitle="Does min-vol's lower realized volatility hold up specifically in stressed markets? Split by real ^GSPC-classified regime."
            >
              {regime ? <SmartBetaRegimeChart rows={regime.performance_by_strategy_and_regime} /> : <LoadingState />}
            </Card>
          </div>
        )}

        {tab === 'capacity' && (
          <div style={{ display: 'grid', gap: 20 }}>
            <Card
              title="Cost-adjusted CAGR vs. AUM (95% CI)"
              subtitle="Real square-root-law rebalancing costs at $10M / $100M / $1B."
            >
              {results ? <CapacityChart rows={results.comparison} /> : <LoadingState />}
            </Card>
            <Card title="Kill-switch status" subtitle="Tracking-error limit (5%) and relative-drawdown limit (10%), AUM=$100M.">
              {killSwitch ? <KillSwitchPanel rows={killSwitch.strategies} /> : <LoadingState />}
            </Card>
            <Card title="Active-risk attribution (Brinson-Fachler)" subtitle="Sector allocation vs. security selection vs. interaction.">
              <AttributionCard data={attribution} />
            </Card>
          </div>
        )}

        {tab === 'results' && (
          <div style={{ display: 'grid', gap: 20 }}>
            <Card title="Honest findings">
              {results && results.findings.length > 0 ? (
                <ul className="findings-list">
                  {results.findings.map((finding, i) => (
                    <li key={i}>{finding}</li>
                  ))}
                </ul>
              ) : (
                <LoadingState />
              )}
            </Card>
            <Card title="Full comparison table">{results ? <ResultsTable rows={results.comparison} /> : <LoadingState />}</Card>
          </div>
        )}
      </main>
    </div>
  )
}
