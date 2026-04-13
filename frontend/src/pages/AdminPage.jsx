import { useEffect, useState } from 'react'
import { apiFetch } from '../api/client'

const DAY_OPTIONS = [7, 14, 30, 90]
const LOW_CONF_PAGE_SIZE = 20

function Section({ title, children }) {
  return (
    <section className="admin-insights-section">
      <h2>{title}</h2>
      {children}
    </section>
  )
}

function Kv({ label, value }) {
  return (
    <div className="admin-insights-kv">
      <span className="admin-insights-kv-label">{label}</span>
      <span className="admin-insights-kv-value">{value}</span>
    </div>
  )
}

function formatPct(v) {
  if (v === null || v === undefined) return '—'
  return `${v}%`
}

function formatDict(obj) {
  if (!obj || typeof obj !== 'object' || Object.keys(obj).length === 0) return '—'
  return Object.entries(obj)
    .map(([k, v]) => `${k}: ${v}`)
    .join(', ')
}

function TokenSparkline({ series }) {
  const pts = series?.days || []
  if (!pts || pts.length < 2) {
    return <p className="muted">Add more days with usage to see a trend.</p>
  }
  const vals = pts.map((d) => Number(d.sum_tokens_estimated) || 0)
  const max = Math.max(...vals, 1)
  const w = 420
  const h = 72
  const pad = 6
  const step = vals.length > 1 ? (w - 2 * pad) / (vals.length - 1) : 0
  const points = vals
    .map((v, i) => {
      const x = pad + i * step
      const y = h - pad - (v / max) * (h - 2 * pad)
      return `${x},${y}`
    })
    .join(' ')
  return (
    <svg
      className="admin-token-sparkline"
      viewBox={`0 0 ${w} ${h}`}
      role="img"
      aria-label="Token usage trend"
    >
      <polyline fill="none" stroke="var(--color-primary)" strokeWidth="2" points={points} />
    </svg>
  )
}

export function AdminPage() {
  const [days, setDays] = useState(7)
  const [lcOffset, setLcOffset] = useState(0)
  const [data, setData] = useState(null)
  const [lowConf, setLowConf] = useState(null)
  const [chunks, setChunks] = useState(null)
  const [tokensByDay, setTokensByDay] = useState(null)
  const [costSummary, setCostSummary] = useState(null)
  const [contentQuality, setContentQuality] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [lcLoading, setLcLoading] = useState(false)
  const [showRaw, setShowRaw] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    const q = `days=${days}`
    ;(async () => {
      try {
        const [summary, ch, tbd, cost, cq] = await Promise.all([
          apiFetch(`/api/admin/insights?${q}`),
          apiFetch(`/api/admin/insights/chunks?${q}&limit=15`),
          apiFetch(`/api/admin/insights/tokens-by-day?${q}`),
          apiFetch(`/api/admin/insights/cost-summary?${q}`),
          apiFetch(`/api/admin/insights/content-quality?${q}`),
        ])
        if (!cancelled) {
          setData(summary)
          setChunks(ch)
          setTokensByDay(tbd)
          setCostSummary(cost)
          setContentQuality(cq)
        }
      } catch (e) {
        if (!cancelled) setError(e.data?.error || e.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [days])

  useEffect(() => {
    let cancelled = false
    setLcLoading(true)
    const q = `days=${days}`
    ;(async () => {
      try {
        const lc = await apiFetch(
          `/api/admin/insights/low-confidence?${q}&limit=${LOW_CONF_PAGE_SIZE}&offset=${lcOffset}`,
        )
        if (!cancelled) setLowConf(lc)
      } catch (e) {
        if (!cancelled) setError(e.data?.error || e.message)
      } finally {
        if (!cancelled) setLcLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [days, lcOffset])

  const w = data?.window
  const vol = data?.volume
  const ret = data?.retrieval
  const pipe = data?.pipeline
  const boost = data?.boost
  const fb = data?.feedback
  const out = data?.outcomes
  const mt = data?.models_and_tokens

  const csvHref = `/api/admin/insights/low-confidence.csv?days=${days}`

  const lcLimit = lowConf?.limit ?? LOW_CONF_PAGE_SIZE
  const lcTotal = lowConf?.total ?? 0
  const canPrevLc = lcOffset > 0
  const canNextLc = lowConf != null && lcTotal > lcOffset + lcLimit

  return (
    <div className="center-page wide">
      <h1>Admin insights</h1>
      <p className="muted">
        UTC aggregates, low-confidence drill-down (no user emails), chunk frequency, and model/token
        rollups when usage is persisted on assistant turns.
      </p>

      <div className="admin-insights-toolbar">
        <label>
          Window (days){' '}
          <select
            value={days}
            onChange={(e) => {
              setDays(Number(e.target.value))
              setLcOffset(0)
            }}
            aria-label="Time window in days"
          >
            {DAY_OPTIONS.map((d) => (
              <option key={d} value={d}>
                Last {d} days
              </option>
            ))}
          </select>
        </label>
        {w ? (
          <span className="muted">
            {w.since} → {w.until} ({w.timezone_note})
          </span>
        ) : null}
        <a className="muted" href={csvHref} download>
          Download low-confidence CSV
        </a>
      </div>

      {error ? <p className="error">{error}</p> : null}
      {loading ? <p className="muted">Loading…</p> : null}

      {data && !loading ? (
        <>
          {data.insufficient_data ? (
            <p className="muted">
              No retrieval events in this window yet — metrics will populate as users chat.
            </p>
          ) : null}

          <Section title="Volume">
            <Kv label="Retrieval events" value={vol?.retrieval_events ?? '—'} />
            <Kv label="Distinct sessions" value={vol?.distinct_sessions ?? '—'} />
          </Section>

          <Section title="Models and tokens">
            <Kv label="Response variants (window)" value={mt?.response_variants_in_window ?? '—'} />
            <Kv
              label="Sum total tokens (OpenAI + Gemini usage, estimated)"
              value={mt?.sum_total_tokens_estimated ?? '—'}
            />
            <Kv label="Variants with token totals" value={mt?.response_variants_with_token_totals ?? '—'} />
            <Kv label="By provider" value={formatDict(mt?.by_provider)} />
            <Kv label="By primary model name" value={formatDict(mt?.by_primary_model_name)} />
          </Section>

          {tokensByDay ? (
            <Section title="Token usage by day (UTC)">
              <p className="muted">
                {tokensByDay.window?.timezone_note ?? 'Per calendar day from response variant timestamps.'}
              </p>
              {tokensByDay.days && tokensByDay.days.length > 0 ? (
                <table className="admin-insights-table">
                  <thead>
                    <tr>
                      <th scope="col">Date (UTC)</th>
                      <th scope="col">Response variants</th>
                      <th scope="col">Sum tokens (est.)</th>
                      <th scope="col">Variants with usage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tokensByDay.days.map((row) => (
                      <tr key={row.date}>
                        <td>{row.date}</td>
                        <td>{row.response_variants}</td>
                        <td>{row.sum_tokens_estimated ?? '—'}</td>
                        <td>{row.variants_with_token_totals}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="muted">No response variants in this window.</p>
              )}
              <h3 className="admin-insights-sub">Trend</h3>
              <TokenSparkline series={tokensByDay} />
            </Section>
          ) : null}

          {costSummary ? (
            <Section title="Cost & budget (estimated)">
              <p className="muted">
                {costSummary.usd_assumption_note ?? 'Set LLM_COST_USD_PER_MTOKENS for USD estimates.'}
              </p>
              <Kv label="Sum tokens (window)" value={costSummary.sum_tokens_estimated ?? '—'} />
              <Kv label="Cap (LLM_MONTHLY_TOKEN_CAP)" value={costSummary.cap_tokens ?? '—'} />
              <Kv label="Warn threshold" value={costSummary.warn_threshold_tokens ?? '—'} />
              <Kv label="Over cap" value={costSummary.over_cap ? 'yes' : 'no'} />
              <Kv label="Near warn" value={costSummary.near_warn_threshold ? 'yes' : 'no'} />
              <Kv label="Estimated USD (blended)" value={costSummary.estimated_usd ?? '—'} />
              <Kv label="Spike note" value={costSummary.spike_note ?? '—'} />
            </Section>
          ) : null}

          <Section title="Retrieval">
            <Kv
              label="Avg confidence"
              value={
                ret?.avg_confidence != null && Number.isFinite(Number(ret.avg_confidence))
                  ? Number(ret.avg_confidence).toFixed(3)
                  : '—'
              }
            />
            <Kv
              label="Avg latency (ms)"
              value={
                ret?.avg_latency_ms != null && Number.isFinite(Number(ret.avg_latency_ms))
                  ? Number(ret.avg_latency_ms).toFixed(0)
                  : '—'
              }
            />
            <Kv label="% no chunks / off-topic" value={formatPct(ret?.pct_no_chunks_or_off_topic)} />
            <Kv label="% low-confidence flag" value={formatPct(ret?.pct_low_confidence_flag)} />
          </Section>

          <Section title="Pipeline">
            <Kv label="Query type (v2)" value={formatDict(pipe?.by_query_type_v2)} />
            <Kv label="Answer mode" value={formatDict(pipe?.by_answer_mode)} />
            <Kv label="Validation passed" value={formatDict(pipe?.validation_passed)} />
            <Kv label="Validation severity (JSON)" value={formatDict(pipe?.validation_severity)} />
          </Section>

          <Section title="Boost">
            <Kv label="Response variants (window)" value={boost?.response_variants_in_window ?? '—'} />
            <Kv label="% boost used" value={formatPct(boost?.pct_boost_used)} />
            <Kv label="By boost_reason" value={formatDict(boost?.by_boost_reason)} />
          </Section>

          <Section title="Feedback">
            <Kv label="Feedback rows" value={fb?.rows ?? '—'} />
            <Kv label="Course thumb" value={formatDict(fb?.course_thumb)} />
            <Kv label="Avg helpfulness (1–5)" value={fb?.avg_helpfulness_rating ?? '—'} />
          </Section>

          <Section title="Outcomes">
            <Kv label="Outcome rows" value={out?.rows ?? '—'} />
            <Kv
              label="% answer_resolved (where known)"
              value={formatPct(out?.pct_answer_resolved_true)}
            />
          </Section>

          {lowConf ? (
            <Section title="Low-confidence drill-down (paged)">
              <p className="muted">
                Total matching: {lowConf.total}. Showing {lowConf.items?.length ?? 0} (limit{' '}
                {lowConf.limit}, offset {lowConf.offset}). Session/message IDs only — no user emails.
              </p>
              <div className="admin-insights-pager">
                <button
                  type="button"
                  disabled={!canPrevLc || lcLoading}
                  onClick={() => setLcOffset((o) => Math.max(0, o - lcLimit))}
                  aria-label="Previous low-confidence page"
                >
                  Previous
                </button>
                <button
                  type="button"
                  disabled={!canNextLc || lcLoading}
                  onClick={() => setLcOffset((o) => o + lcLimit)}
                  aria-label="Next low-confidence page"
                >
                  Next
                </button>
                {lcLoading ? <span className="muted">Loading…</span> : null}
              </div>
              {lowConf.items && lowConf.items.length > 0 ? (
                <ul className="admin-insights-list">
                  {lowConf.items.map((row) => (
                    <li key={row.retrieval_log_id}>
                      <strong>{row.confidence?.toFixed?.(3) ?? '—'}</strong> ·{' '}
                      <span className="muted">{row.query_type_v2 ?? '—'}</span>
                      <div className="admin-insights-snippet">{row.user_question}</div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">No low-confidence rows in this window.</p>
              )}
            </Section>
          ) : null}

          {contentQuality ? (
            <Section title="Content quality (heuristic)">
              <p className="muted">
                Chunks flagged often in low-confidence retrievals; thumbs-down count in window.
              </p>
              <Kv label="Course thumb down (count)" value={contentQuality.course_thumb_down_count ?? '—'} />
              {contentQuality.weak_chunks_by_low_confidence_hits &&
              contentQuality.weak_chunks_by_low_confidence_hits.length > 0 ? (
                <ul className="admin-insights-list">
                  {contentQuality.weak_chunks_by_low_confidence_hits.map((row) => (
                    <li key={row.lecture_chunk_id}>
                      <strong>L{row.lecture_number ?? '?'}</strong> · low-conf hits {row.low_confidence_hit_count}
                      {row.topic ? <span> — {row.topic}</span> : null}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">No weak-chunk signal in this window.</p>
              )}
            </Section>
          ) : null}

          {chunks ? (
            <Section title="Chunk analytics">
              <h3 className="admin-insights-sub">Top chunks in low-confidence retrievals</h3>
              <ChunkList rows={chunks.top_chunks_in_low_confidence_retrievals} />
              <h3 className="admin-insights-sub">Top chunks overall</h3>
              <ChunkList rows={chunks.top_chunks_overall} />
            </Section>
          ) : null}

          <p>
            <button type="button" className="link-btn" onClick={() => setShowRaw((s) => !s)}>
              {showRaw ? 'Hide' : 'Show'} raw JSON (summary)
            </button>
          </p>
          {showRaw ? <pre className="json-preview">{JSON.stringify(data, null, 2)}</pre> : null}
        </>
      ) : null}
    </div>
  )
}

function ChunkList({ rows }) {
  if (!rows || rows.length === 0) {
    return <p className="muted">No chunk hit data in this window.</p>
  }
  return (
    <ul className="admin-insights-list">
      {rows.map((r) => (
        <li key={`${r.lecture_chunk_id}-${r.hit_count}`}>
          <strong>L{r.lecture_number ?? '?'}</strong> · hits {r.hit_count}
          {r.topic ? <span> — {r.topic}</span> : null}
          {r.chunk_key ? <span className="muted"> ({r.chunk_key})</span> : null}
        </li>
      ))}
    </ul>
  )
}
