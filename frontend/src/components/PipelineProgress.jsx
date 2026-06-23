const STEPS = [
  { key: 'parsing',   label: 'Parse',   desc: 'Drain3 template extraction' },
  { key: 'detecting', label: 'Detect',  desc: 'Isolation Forest anomaly detection' },
  { key: 'embedding', label: 'Embed',   desc: 'Sentence-transformer + FAISS indexing' },
  { key: 'ready',     label: 'Ready',   desc: 'RAG pipeline loaded' },
]

const M2_STAGE_LABEL = {
  loading:      'Loading events from CSV',
  sessionizing: 'Grouping events into sessions',
  vectorizing:  'Building event-count feature vectors',
  training:     'Training Isolation Forest',
  scoring:      'Scoring sessions for anomalies',
}

function stepIndex(step) {
  const idx = STEPS.findIndex(s => s.key === step)
  return idx === -1 ? (step === 'idle' ? -1 : STEPS.length) : idx
}

function fmt(n) { return n?.toLocaleString() ?? '—' }

/* ── Module 1 detail ──────────────────────────────────────────────────── */
function ParsingDetail({ s }) {
  const { parsing_total: total, parsing_scanned: scanned,
          parsing_kept: kept, parsing_templates: templates,
          parsing_rate: rate, parsing_current_line: currentLine } = s
  if (!total) return null

  const pending = Math.max(0, total - (scanned ?? 0))

  return (
    <div className="mt-2 pl-1 space-y-1 font-mono text-xs leading-relaxed select-none">
      <div className="flex flex-wrap gap-x-5 gap-y-0.5 text-gray-600">
        <span><span className="text-gray-500">processed</span> <span className="text-gray-400">{fmt(scanned)}</span></span>
        <span><span className="text-gray-500">pending</span>   <span className="text-gray-400">{fmt(pending)}</span></span>
        <span><span className="text-gray-500">kept</span>      <span className="text-gray-400">{fmt(kept)}</span></span>
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-0.5 text-gray-600">
        <span><span className="text-gray-500">templates</span> <span className="text-gray-400">{fmt(templates)}</span></span>
        {rate > 0 && <span><span className="text-gray-500">rate</span> <span className="text-gray-400">{fmt(rate)}/s</span></span>}
      </div>
      {currentLine && (
        <div className="mt-1 pt-1 border-t border-gray-700/50 overflow-hidden">
          <span className="text-gray-700">→ </span>
          <span className="text-gray-700 truncate block" style={{ maxWidth: '100%' }}>{currentLine}</span>
        </div>
      )}
    </div>
  )
}

/* ── Module 2 detail ──────────────────────────────────────────────────── */
function DetectingDetail({ s }) {
  const label = M2_STAGE_LABEL[s.m2_stage]
  if (!label) return null
  return (
    <div className="mt-2 pl-1 font-mono text-xs text-gray-600 select-none">
      <span className="text-gray-500">stage </span>
      <span className="text-gray-400">{label}</span>
    </div>
  )
}

/* ── Module 3 detail ──────────────────────────────────────────────────── */
function EmbeddingDetail({ s }) {
  const { embed_done: done, embed_total: total } = s
  if (!total) return null
  const pct = Math.round((done / total) * 100)
  return (
    <div className="mt-2 pl-1 space-y-1 font-mono text-xs text-gray-600 select-none">
      <div className="flex gap-x-5">
        <span><span className="text-gray-500">embedded</span> <span className="text-gray-400">{fmt(done)}</span></span>
        <span><span className="text-gray-500">remaining</span> <span className="text-gray-400">{fmt(total - done)}</span></span>
        <span><span className="text-gray-500">total</span> <span className="text-gray-400">{fmt(total)}</span></span>
      </div>
      {total > 0 && (
        <div className="w-full h-1 bg-gray-700 rounded-full overflow-hidden">
          <div className="h-full bg-blue-700 rounded-full transition-all duration-300" style={{ width: `${pct}%` }} />
        </div>
      )}
    </div>
  )
}

/* ── Main component ───────────────────────────────────────────────────── */
export default function PipelineProgress({ status }) {
  const { step = 'idle', message = '', progress_pct = 0, stats } = status || {}
  const current = stepIndex(step)
  const isError = step === 'error'
  const isReady = step === 'ready'

  return (
    <div className="card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">Pipeline Progress</h3>
        {isError && <span className="text-xs text-red-400 bg-red-900/30 px-2 py-0.5 rounded-full">Error</span>}
        {isReady && <span className="text-xs text-green-400 bg-green-900/30 px-2 py-0.5 rounded-full">Complete</span>}
      </div>

      {/* Global progress bar */}
      <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${isError ? 'bg-red-500' : 'bg-blue-500'}`}
          style={{ width: `${progress_pct}%` }}
        />
      </div>

      {/* Step indicators */}
      <div className="flex justify-between">
        {STEPS.map((s, i) => {
          const done   = i < current || isReady
          const active = i === current && !isReady && !isError
          return (
            <div key={s.key} className="flex flex-col items-center gap-1.5 flex-1">
              <div className={`
                w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold transition-colors
                ${done   ? 'bg-blue-600 text-white' : ''}
                ${active && !isError ? 'bg-blue-900 text-blue-300 ring-2 ring-blue-500' : ''}
                ${active && isError  ? 'bg-red-900 text-red-300 ring-2 ring-red-500'   : ''}
                ${!done && !active  ? 'bg-gray-700 text-gray-500' : ''}
              `}>
                {done ? '✓' : i + 1}
              </div>
              <span className={`text-xs text-center ${done ? 'text-blue-400' : active ? 'text-gray-200' : 'text-gray-600'}`}>
                {s.label}
              </span>
            </div>
          )
        })}
      </div>

      {/* Status message + per-module live detail */}
      <div>
        <p className={`text-xs ${isError ? 'text-red-400' : isReady ? 'text-green-400' : 'text-gray-400'}`}>
          {message}
        </p>
        {step === 'parsing'   && <ParsingDetail   s={status} />}
        {step === 'detecting' && <DetectingDetail s={status} />}
        {step === 'embedding' && <EmbeddingDetail s={status} />}
      </div>

      {/* Completion summary */}
      {isReady && stats && (
        <div className="grid grid-cols-3 gap-3 pt-1 border-t border-gray-700">
          {[
            { label: 'Sessions',  value: stats.total_sessions?.toLocaleString() },
            { label: 'Anomalies', value: stats.anomalous_sessions?.toLocaleString() },
            { label: 'Indexed',   value: stats.index_size?.toLocaleString() },
          ].map(({ label, value }) => (
            <div key={label} className="text-center">
              <div className="text-base font-bold text-blue-400">{value ?? '—'}</div>
              <div className="text-xs text-gray-500">{label}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
