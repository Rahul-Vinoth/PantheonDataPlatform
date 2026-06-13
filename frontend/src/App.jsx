import { useEffect, useRef, useState } from 'react'

const API = '/api'

const get  = url => fetch(url).then(r => r.json())
const post = (url, body) =>
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify(body) }).then(r => r.json())

const C = {
  bg:       '#f1f5f9',
  surface:  '#ffffff',
  border:   '#e2e8f0',
  accent:   '#6366f1',
  accentHo: '#4f46e5',
  ok:       '#16a34a',
  err:      '#dc2626',
  warn:     '#d97706',
  muted:    '#94a3b8',
  text:     '#0f172a',
  dim:      '#475569',
}

const S = {
  app: { minHeight: '100vh', background: C.bg, color: C.text,
         fontFamily: '"Inter", system-ui, sans-serif', padding: '24px 32px' },
  header: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 32 },
  logo: { width: 36, height: 36, background: C.accent, borderRadius: 8,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 18, fontWeight: 700, color: '#fff' },
  title: { fontSize: 20, fontWeight: 700, margin: 0 },
  subtitle: { fontSize: 13, color: C.muted, margin: '2px 0 0' },
  grid: { display: 'grid', gridTemplateColumns: '280px 1fr', gap: 20,
          alignItems: 'start' },
  card: { background: C.surface, border: `1px solid ${C.border}`,
          borderRadius: 12, overflow: 'hidden' },
  cardHead: { padding: '14px 18px', borderBottom: `1px solid ${C.border}`,
              fontWeight: 600, fontSize: 13, letterSpacing: '0.05em',
              textTransform: 'uppercase', color: C.dim },
  cardBody: { padding: 18 },
  btn: { display: 'inline-flex', alignItems: 'center', gap: 6,
         padding: '6px 14px', borderRadius: 6, border: 'none',
         background: C.accent, color: '#fff', fontSize: 13, fontWeight: 500,
         cursor: 'pointer' },
  btnSm: { padding: '4px 10px', fontSize: 12 },
  pill: { display: 'inline-block', padding: '2px 8px', borderRadius: 99,
          fontSize: 11, fontWeight: 600 },
  terminal: { background: '#1e293b', borderRadius: 8, padding: 14,
              fontFamily: '"JetBrains Mono", "Fira Code", monospace',
              fontSize: 12, lineHeight: 1.6, maxHeight: 420,
              overflowY: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
              color: '#86efac' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th: { textAlign: 'left', padding: '8px 10px', color: C.muted,
        borderBottom: `1px solid ${C.border}`, fontWeight: 500, fontSize: 12 },
  td: { padding: '8px 10px', borderBottom: `1px solid ${C.border}` },
  tag: { background: '#ede9fe', color: '#4f46e5', padding: '2px 7px',
         borderRadius: 4, fontSize: 11, fontWeight: 600 },
}

function StatusPill({ status }) {
  const map = {
    idle:    [C.muted,  '#f1f5f9', 'IDLE'],
    running: [C.warn,   '#fef3c7', 'RUNNING'],
    done:    [C.ok,     '#dcfce7', 'DONE'],
    error:   [C.err,    '#fee2e2', 'ERROR'],
  }
  const [color, bg, label] = map[status] ?? [C.muted, '#1f2937', status]
  return <span style={{ ...S.pill, color, background: bg }}>{label}</span>
}

// ── Sources ──────────────────────────────────────────────────────────────────
function Sources({ onIngest, jobStatus }) {
  const [sources, setSources] = useState([])
  const busy = jobStatus === 'running'

  useEffect(() => {
    const load = () => get(`${API}/sources`).then(d => setSources(d.sources))
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [])

  return (
    <div style={S.card}>
      <div style={S.cardHead}>Data Sources</div>
      <div style={S.cardBody}>
        <p style={{ color: C.dim, fontSize: 12, margin: '0 0 14px' }}>
          Run the pipeline (<b>ingest → QC → encode → IDM</b>) on a folder in{' '}
          <code style={{ color: C.accentHo }}>realdata/</code>
        </p>
        {sources.length === 0
          ? <p style={{ color: C.muted, fontSize: 13 }}>
              No sources found. Add a folder to <code>realdata/</code>.
            </p>
          : sources.map(s => (
            <div key={s.name} style={{ display: 'flex', alignItems: 'center',
                                        justifyContent: 'space-between', marginBottom: 10,
                                        padding: '10px 12px', background: C.bg,
                                        borderRadius: 8, border: `1px solid ${C.border}` }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{s.name}</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                  {s.file_count} files
                </div>
              </div>
              <button
                style={{ ...S.btn, ...S.btnSm, opacity: busy ? 0.5 : 1,
                         cursor: busy ? 'not-allowed' : 'pointer' }}
                disabled={busy}
                onClick={() => onIngest(s.name)}
              >
                {busy ? '…' : 'Run'}
              </button>
            </div>
          ))
        }
      </div>
    </div>
  )
}

// ── Ingest Log ────────────────────────────────────────────────────────────────
function StageTracker({ stages, stage, jobStatus }) {
  const activeIdx = stage ? stages.indexOf(stage) : (jobStatus === 'done' ? stages.length : -1)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 14 }}>
      {stages.map((st, i) => {
        const done    = i < activeIdx || jobStatus === 'done'
        const current = i === activeIdx && jobStatus === 'running'
        const failed  = i === activeIdx && jobStatus === 'error'
        const color = failed ? C.err : current ? C.warn : done ? C.ok : C.muted
        const bg    = failed ? '#fee2e2' : current ? '#fef3c7' : done ? '#dcfce7' : '#f1f5f9'
        return (
          <span key={st} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ ...S.pill, color, background: bg,
                           display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              {done ? '✓' : current ? '●' : '○'} {st}
            </span>
            {i < stages.length - 1 &&
              <span style={{ color: C.muted, fontSize: 12 }}>→</span>}
          </span>
        )
      })}
    </div>
  )
}

function IngestLog({ jobStatus, triggerKey, stage, stages }) {
  const [lines, setLines] = useState([])
  const termRef = useRef(null)
  const esRef   = useRef(null)

  useEffect(() => {
    if (!triggerKey) return
    setLines([])
    if (esRef.current) esRef.current.close()

    const es = new EventSource(`${API}/ingest/stream`)
    esRef.current = es
    es.onmessage = e => {
      const line = JSON.parse(e.data)
      if (line === '__done__') { es.close(); return }
      setLines(prev => [...prev, line])
    }
    return () => es.close()
  }, [triggerKey])

  useEffect(() => {
    if (termRef.current)
      termRef.current.scrollTop = termRef.current.scrollHeight
  }, [lines])

  return (
    <div style={S.card}>
      <div style={{ ...S.cardHead, display: 'flex', alignItems: 'center',
                    justifyContent: 'space-between' }}>
        <span>Pipeline Log</span>
        <StatusPill status={jobStatus} />
      </div>
      <div style={S.cardBody}>
        <StageTracker stages={stages} stage={stage} jobStatus={jobStatus} />
        <div style={S.terminal} ref={termRef}>
          {lines.length === 0
            ? <span style={{ color: C.muted }}>Waiting for ingest to start…</span>
            : lines.map((l, i) => <div key={i}>{l || ' '}</div>)
          }
        </div>
      </div>
    </div>
  )
}

// ── Lakehouse ─────────────────────────────────────────────────────────────────
function QueryBlock({ name, rows }) {
  const [open, setOpen] = useState(true)
  const title = name.replace(/_/g, ' ')

  const headBtn = (
    <button onClick={() => setOpen(o => !o)}
            style={{ background: 'none', border: 'none', color: C.dim,
                     fontSize: 12, fontWeight: 600, cursor: 'pointer',
                     textTransform: 'uppercase', letterSpacing: '0.05em',
                     padding: 0, marginBottom: 8 }}>
      {open ? '▾' : '▸'} {title}
      {Array.isArray(rows) &&
        <span style={{ color: C.muted, fontWeight: 400, marginLeft: 6 }}>
          ({rows.length} rows)
        </span>
      }
    </button>
  )

  if (!Array.isArray(rows) || rows.length === 0)
    return (
      <div style={{ marginBottom: 16 }}>
        {headBtn}
        {open && <p style={{ color: C.muted, fontSize: 13, margin: '4px 0 0 10px' }}>No rows.</p>}
      </div>
    )

  const cols = Object.keys(rows[0])
  return (
    <div style={{ marginBottom: 20 }}>
      {headBtn}
      {open && (
        <table style={S.table}>
          <thead>
            <tr>{cols.map(c => <th key={c} style={S.th}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {cols.map(c => {
                  const val = r[c]
                  return (
                    <td key={c} style={S.td}>
                      {typeof val === 'number'
                        ? <span style={{ color: C.accent, fontWeight: 600 }}>{val}</span>
                        : String(val ?? '—').length > 24
                          ? <span style={S.tag} title={String(val)}>{String(val).slice(0, 16)}…</span>
                          : <span style={S.tag}>{String(val ?? '—')}</span>
                      }
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function LakehouseView({ refreshKey }) {
  const [data, setData] = useState(null)
  const load = () => get(`${API}/lakehouse`).then(setData)

  useEffect(() => { load() }, [refreshKey])

  if (!data)
    return (
      <div style={{ ...S.card, marginTop: 20 }}>
        <div style={S.cardHead}>Lakehouse</div>
        <div style={{ ...S.cardBody, color: C.muted }}>Loading…</div>
      </div>
    )

  const { tables, queries } = data

  return (
    <div style={{ ...S.card, marginTop: 20 }}>
      <div style={{ ...S.cardHead, display: 'flex', alignItems: 'center',
                    justifyContent: 'space-between' }}>
        <span>Lakehouse State</span>
        <button style={{ ...S.btn, ...S.btnSm }} onClick={load}>Refresh</button>
      </div>
      <div style={S.cardBody}>
        <p style={{ color: C.dim, fontSize: 12, margin: '0 0 10px', fontWeight: 600,
                    textTransform: 'uppercase', letterSpacing: '0.05em' }}>Tables</p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 24 }}>
          {Object.entries(tables).map(([name, count]) => (
            <div key={name} style={{ background: C.bg, border: `1px solid ${C.border}`,
                                      borderRadius: 8, padding: '8px 12px', minWidth: 110 }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 2 }}>{name}</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: C.accent }}>{count}</div>
              <div style={{ fontSize: 10, color: C.muted }}>rows</div>
            </div>
          ))}
          {Object.keys(tables).length === 0 &&
            <p style={{ color: C.muted, fontSize: 13 }}>No tables yet — run an ingest first.</p>
          }
        </div>

        {Object.entries(queries).map(([name, rows]) => (
          <QueryBlock key={name} name={name} rows={rows} />
        ))}
      </div>
    </div>
  )
}

// ── Delivery / Export ───────────────────────────────────────────────────────
function Toggle({ label, checked, onChange }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13,
                    cursor: 'pointer', marginBottom: 6 }}>
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
      {label}
    </label>
  )
}

function DeliveryPanel({ refreshKey }) {
  const [opts, setOpts]       = useState(null)
  const [sel, setSel]         = useState({})         // source_id -> bool
  const [name, setName]       = useState('delivery_1')
  const [withEmb, setWithEmb] = useState(true)
  const [withAct, setWithAct] = useState(true)
  const [okOnly, setOkOnly]   = useState(false)
  const [media, setMedia]     = useState(true)
  const [status, setStatus]   = useState('idle')
  const [lines, setLines]     = useState([])
  const [exports, setExports] = useState([])
  const esRef = useRef(null)

  const loadOpts    = () => get(`${API}/export/options`).then(setOpts)
  const loadExports = () => get(`${API}/exports`).then(d => setExports(d.exports))
  useEffect(() => { loadOpts(); loadExports() }, [refreshKey])

  const chosenSources = () => Object.keys(sel).filter(s => sel[s])

  const run = async () => {
    setLines([]); setStatus('running')
    const res = await post(`${API}/export`, {
      name,
      sources: chosenSources().length ? chosenSources() : null,
      include_partial: !okOnly,
      include_embeddings: withEmb,
      include_action_latents: withAct,
      copy_media: media,
    })
    if (res.status !== 'started') { alert(res.detail ?? 'failed'); setStatus('error'); return }
    if (esRef.current) esRef.current.close()
    const es = new EventSource(`${API}/export/stream`); esRef.current = es
    es.onmessage = e => {
      const l = JSON.parse(e.data)
      if (l === '__done__') { es.close(); return }
      setLines(p => [...p, l])
    }
    const poll = setInterval(async () => {
      const s = await get(`${API}/export/status`)
      setStatus(s.status)
      if (s.status !== 'running') { clearInterval(poll); loadExports() }
    }, 800)
  }

  return (
    <div style={{ ...S.card, marginTop: 20 }}>
      <div style={{ ...S.cardHead, display: 'flex', alignItems: 'center',
                    justifyContent: 'space-between' }}>
        <span>Delivery — Package for Training</span>
        <StatusPill status={status} />
      </div>
      <div style={S.cardBody}>
        <p style={{ color: C.dim, fontSize: 12, margin: '0 0 14px' }}>
          Select what to package into a portable, Lance-decoupled bundle (Parquet + media
          + manifest). Manual — not part of the pipeline.
        </p>

        {/* sources */}
        <p style={{ fontSize: 12, fontWeight: 600, color: C.dim, margin: '0 0 6px',
                    textTransform: 'uppercase', letterSpacing: '0.05em' }}>Sources</p>
        {opts?.sources?.length
          ? opts.sources.map(s => (
              <Toggle key={s.source_id}
                      label={`${s.name} (${s.episodes} episodes)`}
                      checked={!!sel[s.source_id]}
                      onChange={v => setSel(p => ({ ...p, [s.source_id]: v }))} />
            ))
          : <p style={{ color: C.muted, fontSize: 13 }}>No episodes yet.</p>}
        <p style={{ fontSize: 11, color: C.muted, margin: '2px 0 14px' }}>
          (none selected = all sources)
        </p>

        {/* options */}
        <p style={{ fontSize: 12, fontWeight: 600, color: C.dim, margin: '0 0 6px',
                    textTransform: 'uppercase', letterSpacing: '0.05em' }}>Include</p>
        <Toggle label="Embeddings" checked={withEmb} onChange={setWithEmb} />
        <Toggle label="Action latents (IDM)" checked={withAct} onChange={setWithAct} />
        <Toggle label="Copy media into bundle (self-contained)" checked={media} onChange={setMedia} />
        <Toggle label="OK episodes only (exclude partial)" checked={okOnly} onChange={setOkOnly} />

        {/* name + run */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 12 }}>
          <input value={name} onChange={e => setName(e.target.value)}
                 placeholder="bundle name"
                 style={{ flex: 1, padding: '6px 10px', borderRadius: 6,
                          border: `1px solid ${C.border}`, fontSize: 13,
                          background: C.bg, color: C.text }} />
          <button style={{ ...S.btn, opacity: status === 'running' ? 0.5 : 1,
                           cursor: status === 'running' ? 'not-allowed' : 'pointer' }}
                  disabled={status === 'running'} onClick={run}>
            {status === 'running' ? 'Packaging…' : 'Package delivery'}
          </button>
        </div>

        {/* log */}
        {lines.length > 0 &&
          <div style={{ ...S.terminal, marginTop: 12, maxHeight: 160 }}>
            {lines.map((l, i) => <div key={i}>{l || ' '}</div>)}
          </div>}

        {/* existing bundles */}
        {exports.length > 0 && <>
          <p style={{ fontSize: 12, fontWeight: 600, color: C.dim, margin: '18px 0 6px',
                      textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Delivery bundles
          </p>
          {exports.map(m => (
            <div key={m.name} style={{ background: C.bg, border: `1px solid ${C.border}`,
                                       borderRadius: 8, padding: '10px 12px', marginBottom: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13 }}>{m.name}</div>
              <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                {m.counts.episode} episodes · {m.counts.embedding} emb · {m.counts.action_latent} latents
                · {m.counts.media_files} media · {m.format}
              </div>
              <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>
                enc: {m.encoder_versions.join(',') || '—'} · idm: {m.idm_versions.join(',') || '—'}
              </div>
            </div>
          ))}
        </>}
      </div>
    </div>
  )
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [jobStatus, setJobStatus]   = useState('idle')
  const [stage, setStage]           = useState(null)
  const [stages, setStages]         = useState(['ingest', 'qc', 'encode', 'idm'])
  const [triggerKey, setTriggerKey] = useState(null)
  const [lhRefresh, setLhRefresh]   = useState(0)

  useEffect(() => {
    const t = setInterval(async () => {
      const s = await get(`${API}/ingest/status`)
      setJobStatus(s.status)
      setStage(s.stage)
      if (s.stages) setStages(s.stages)
      if (s.status === 'done' || s.status === 'error')
        setLhRefresh(n => n + 1)
    }, 1000)
    return () => clearInterval(t)
  }, [])

  const handleIngest = async source => {
    const res = await post(`${API}/ingest`, { source })
    if (res.status === 'started') {
      setTriggerKey(Date.now())
      setJobStatus('running')
    } else {
      alert(res.detail ?? 'Failed to start ingest')
    }
  }

  return (
    <div style={S.app}>
      <header style={S.header}>
        <div style={S.logo}>P</div>
        <div>
          <h1 style={S.title}>Pantheon Data Platform</h1>
          <p style={S.subtitle}>Robot foundation model data — ingest · normalize · query</p>
        </div>
      </header>

      <div style={S.grid}>
        <Sources onIngest={handleIngest} jobStatus={jobStatus} />
        <div>
          <IngestLog jobStatus={jobStatus} triggerKey={triggerKey}
                     stage={stage} stages={stages} />
          <LakehouseView refreshKey={lhRefresh} />
          <DeliveryPanel refreshKey={lhRefresh} />
        </div>
      </div>
    </div>
  )
}
