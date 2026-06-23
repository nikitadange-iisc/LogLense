import { Fragment } from 'react'

const STEPS = [
  { key: 'parsing',   label: 'Parse',   desc: 'Drain3 template extraction' },
  { key: 'detecting', label: 'Detect',  desc: 'Isolation Forest anomaly detection' },
  { key: 'embedding', label: 'Embed',   desc: 'Sentence-transformer + FAISS' },
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

/* ── Step icons ───────────────────────────────────────────────────────────── */
function StepIcon({ stepKey }) {
  const cls = 'w-4 h-4'
  switch (stepKey) {
    case 'parsing':
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      )
    case 'detecting':
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      )
    case 'embedding':
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
            d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
        </svg>
      )
    case 'ready':
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    default:
      return null
  }
}

function CheckIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
    </svg>
  )
}

function XIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

/* ── Module detail panels ─────────────────────────────────────────────────── */
function ParsingDetail({ s }) {
  const { parsing_total: total, parsing_scanned: scanned,
          parsing_kept: kept, parsing_templates: templates,
          parsing_rate: rate, parsing_current_line: currentLine } = s
  if (!total) return null
  const pending = Math.max(0, total - (scanned ?? 0))
  return (
    <div className="mt-3 pl-1 space-y-1.5 font-mono text-xs leading-relaxed select-none">
      <div className="flex flex-wrap gap-x-5 gap-y-0.5">
        <span><span className="text-gray-500">processed</span> <span className="text-gray-600 dark:text-gray-400 tabular-nums">{fmt(scanned)}</span></span>
        <span><span className="text-gray-500">pending</span>   <span className="text-gray-600 dark:text-gray-400 tabular-nums">{fmt(pending)}</span></span>
        <span><span className="text-gray-500">kept</span>      <span className="text-gray-600 dark:text-gray-400 tabular-nums">{fmt(kept)}</span></span>
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-0.5">
        <span><span className="text-gray-500">templates</span> <span className="text-gray-600 dark:text-gray-400 tabular-nums">{fmt(templates)}</span></span>
        {rate > 0 && <span><span className="text-gray-500">rate</span> <span className="text-gray-600 dark:text-gray-400 tabular-nums">{fmt(rate)}/s</span></span>}
      </div>
      {/* Mini parse progress bar */}
      {total > 0 && (
        <div className="w-full h-0.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-300"
            style={{ width: `${Math.min((scanned / total) * 100, 100)}%` }}
          />
        </div>
      )}
      {currentLine && (
        <div className="mt-1 pt-1 border-t border-gray-100 dark:border-gray-700/50 overflow-hidden">
          <span className="text-gray-400 dark:text-gray-700 truncate block" style={{ maxWidth: '100%' }}>
            → {currentLine}
          </span>
        </div>
      )}
    </div>
  )
}

function DetectingDetail({ s }) {
  const label = M2_STAGE_LABEL[s.m2_stage]
  if (!label) return null
  return (
    <div className="mt-3 pl-1 font-mono text-xs text-gray-500 dark:text-gray-600 select-none flex items-center gap-2">
      <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
      <span className="text-gray-600 dark:text-gray-400">{label}</span>
    </div>
  )
}

function EmbeddingDetail({ s }) {
  const { embed_done: done, embed_total: total } = s
  if (!total) return null
  const pct = Math.round((done / total) * 100)
  return (
    <div className="mt-3 pl-1 space-y-1.5 font-mono text-xs text-gray-500 dark:text-gray-600 select-none">
      <div className="flex gap-x-5">
        <span><span className="text-gray-500">embedded</span>  <span className="text-gray-600 dark:text-gray-400 tabular-nums">{fmt(done)}</span></span>
        <span><span className="text-gray-500">remaining</span> <span className="text-gray-600 dark:text-gray-400 tabular-nums">{fmt(total - done)}</span></span>
        <span><span className="text-gray-500">total</span>     <span className="text-gray-600 dark:text-gray-400 tabular-nums">{fmt(total)}</span></span>
      </div>
      <div className="w-full h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className="h-full bg-blue-600 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

/* ── Main component ───────────────────────────────────────────────────────── */
export default function PipelineProgress({ status }) {
  const { step = 'idle', message = '', progress_pct = 0, stats, failed_at_step } = status || {}
  const current  = stepIndex(step)
  const isError  = step === 'error'
  const isReady  = step === 'ready'
  const isRunning = !isError && !isReady && step !== 'idle'
  const failedAt = isError ? stepIndex(failed_at_step ?? '') : -1

  return (
    <div className="card p-5 space-y-5">

      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Pipeline Progress</h3>
        {isError && (
          <span className="text-xs text-red-400 bg-red-900/30 border border-red-700/30 px-2 py-0.5 rounded-full">
            Failed
          </span>
        )}
        {isReady && (
          <span className="text-xs text-emerald-400 bg-emerald-900/30 border border-emerald-700/30 px-2 py-0.5 rounded-full">
            Complete
          </span>
        )}
        {isRunning && (
          <span className="text-xs text-blue-400 bg-blue-900/20 border border-blue-700/20 px-2 py-0.5 rounded-full flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            Running
          </span>
        )}
      </div>

      {/* Global progress bar */}
      <div className="relative w-full h-2 bg-gray-100 dark:bg-gray-700/60 rounded-full overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 rounded-full transition-all duration-500 ${
            isError  ? 'bg-red-500' :
            isReady  ? 'bg-emerald-500' :
            isRunning ? 'progress-shimmer' :
            'bg-blue-600'
          }`}
          style={{ width: `${Math.max(progress_pct, 0)}%` }}
        />
      </div>

      {/* Step indicators with connecting lines */}
      <div>
        {/* Circle row */}
        <div className="relative flex items-center">
          {/* Track line behind everything */}
          <div className="absolute inset-x-4 top-1/2 -translate-y-1/2 h-0.5 bg-gray-200 dark:bg-gray-700" />
          {/* Progress fill */}
          <div
            className="absolute inset-x-4 top-1/2 -translate-y-1/2 h-0.5 bg-blue-500/60 transition-all duration-500 origin-left"
            style={{
              width: isReady ? 'calc(100% - 2rem)' :
                     isError && failedAt >= 0 ? `calc(${(failedAt / (STEPS.length - 1)) * 100}% - ${failedAt === 0 ? 2 : 0}rem)` :
                     current >= 0 ? `calc(${(current / (STEPS.length - 1)) * 100}% - 2rem)` :
                     '0%',
            }}
          />
          {/* Circles */}
          <div className="relative z-10 flex justify-between w-full">
            {STEPS.map((s, i) => {
              const done     = isError ? (failedAt >= 0 && i < failedAt) : (i < current || isReady)
              const active   = !isError && i === current && !isReady
              const isFailed = isError && i === failedAt
              return (
                <div key={s.key}
                  className={`
                    w-9 h-9 rounded-full flex items-center justify-center
                    transition-all duration-300 border-2
                    ${done     ? 'bg-blue-600 border-blue-500 text-white shadow-md shadow-blue-600/30' : ''}
                    ${isFailed ? 'bg-red-600 border-red-500 text-white shadow-md shadow-red-600/30' : ''}
                    ${active   ? 'bg-gray-900 dark:bg-gray-800 border-blue-500 text-blue-300 step-pulse' : ''}
                    ${!done && !active && !isFailed ? 'bg-gray-100 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-400 dark:text-gray-500' : ''}
                  `}
                >
                  {isFailed ? <XIcon /> : done ? <CheckIcon /> : <StepIcon stepKey={s.key} />}
                </div>
              )
            })}
          </div>
        </div>

        {/* Label row */}
        <div className="flex justify-between mt-2">
          {STEPS.map((s, i) => {
            const done     = isError ? (failedAt >= 0 && i < failedAt) : (i < current || isReady)
            const active   = !isError && i === current && !isReady
            const isFailed = isError && i === failedAt
            return (
              <span key={s.key} className={`w-9 text-[10px] font-medium text-center block transition-colors ${
                isFailed ? 'text-red-500 dark:text-red-400' :
                done      ? 'text-blue-500 dark:text-blue-400' :
                active    ? 'text-gray-800 dark:text-gray-200' :
                            'text-gray-400 dark:text-gray-600'
              }`}>
                {s.label}
              </span>
            )
          })}
        </div>
      </div>

      {/* Status message + per-module detail */}
      <div className="border-t border-gray-100 dark:border-gray-700/50 pt-3">
        <p className={`text-xs leading-relaxed ${
          isError ? 'text-red-400' : isReady ? 'text-emerald-400' : 'text-gray-500 dark:text-gray-400'
        }`}>
          {message}
        </p>
        {step === 'parsing'   && <ParsingDetail   s={status} />}
        {step === 'detecting' && <DetectingDetail s={status} />}
        {step === 'embedding' && <EmbeddingDetail s={status} />}
      </div>

      {/* Completion summary */}
      {isReady && stats && (
        <div className="grid grid-cols-3 gap-3 pt-1 border-t border-gray-100 dark:border-gray-700/50">
          {[
            { label: 'Sessions',  value: stats.total_sessions?.toLocaleString(),     color: 'text-blue-400' },
            { label: 'Anomalies', value: stats.anomalous_sessions?.toLocaleString(), color: 'text-red-400' },
            { label: 'Indexed',   value: stats.index_size?.toLocaleString(),         color: 'text-emerald-400' },
          ].map(({ label, value, color }) => (
            <div key={label} className="text-center">
              <div className={`text-base font-bold ${color}`}>{value ?? '—'}</div>
              <div className="text-xs text-gray-500 dark:text-gray-600 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
