const N_BINS = 28

function buildBins(scores) {
  if (!scores || scores.length === 0) return { bins: [], minS: 0, maxS: 1, range: 1, maxCount: 1 }
  const vals = scores.map(s => s.score)
  const minS = Math.min(...vals)
  const maxS = Math.max(...vals)
  const range = maxS - minS || 1
  const bw = range / N_BINS

  const bins = Array.from({ length: N_BINS }, (_, i) => ({
    lo: minS + i * bw,
    hi: minS + (i + 1) * bw,
    normal: 0, anomalous: 0,
  }))

  scores.forEach(({ score, is_anomalous }) => {
    const idx = Math.min(Math.floor((score - minS) / bw), N_BINS - 1)
    if (is_anomalous) bins[idx].anomalous++
    else bins[idx].normal++
  })

  const maxCount = Math.max(...bins.map(b => b.normal + b.anomalous), 1)
  return { bins, minS, maxS, range, maxCount }
}

export default function ScoreDistributionChart({ scores }) {
  if (!scores || scores.length === 0) {
    return (
      <div className="card p-5 flex flex-col items-center justify-center h-64 gap-3">
        <svg className="w-8 h-8 text-gray-300 dark:text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        <p className="text-sm text-gray-500 dark:text-gray-600">No distribution data for this session.</p>
        <p className="text-xs text-gray-400 dark:text-gray-700 text-center max-w-xs">
          This session was created before score tracking was added.
          Re-upload the log file to generate the chart.
        </p>
      </div>
    )
  }

  const { bins, minS, maxS, range, maxCount } = buildBins(scores)

  // SVG coordinate system
  const W = 580, H = 270
  const PAD = { top: 18, right: 18, bottom: 52, left: 46 }
  const cw = W - PAD.left - PAD.right
  const ch = H - PAD.top - PAD.bottom
  const baseY = PAD.top + ch
  const bw = cw / N_BINS
  const gap = Math.max(0.8, bw * 0.08)

  const toX = (score) => PAD.left + ((score - minS) / range) * cw
  const toH = (count) => (count / maxCount) * ch

  // Decision boundary at score = 0 (if within range)
  const boundaryX = (minS < 0 && maxS > 0) ? toX(0) : null

  // X axis ticks — 5 evenly spaced
  const xTicks = Array.from({ length: 5 }, (_, i) => {
    const score = minS + (i / 4) * range
    return { x: toX(score), label: score.toFixed(2) }
  })

  // Y axis ticks — 3 levels
  const yTicks = [0, Math.round(maxCount / 2), maxCount]

  const nAnomalous = scores.filter(s => s.is_anomalous).length
  const rate = scores.length > 0 ? (nAnomalous / scores.length * 100).toFixed(1) : '0.0'

  return (
    <div className="card p-5 space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Score Distribution</h3>
          <p className="text-xs text-gray-400 dark:text-gray-600 mt-0.5">
            Isolation Forest decision scores — lower = more anomalous
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs shrink-0">
          <span className="flex items-center gap-1.5 text-gray-500 dark:text-gray-500">
            <svg width="12" height="12"><rect width="12" height="12" rx="2" fill="#64748b" fillOpacity="0.55"/></svg>
            Normal
          </span>
          <span className="flex items-center gap-1.5 text-gray-500 dark:text-gray-500">
            <svg width="12" height="12"><rect width="12" height="12" rx="2" fill="#ef4444" fillOpacity="0.85"/></svg>
            Anomalous
          </span>
          {boundaryX && (
            <span className="flex items-center gap-1.5 text-gray-500 dark:text-gray-500">
              <svg width="12" height="12">
                <line x1="6" y1="0" x2="6" y2="12" stroke="#f97316" strokeWidth="1.5" strokeDasharray="2,2"/>
              </svg>
              Boundary
            </span>
          )}
        </div>
      </div>

      {/* SVG Chart */}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ height: '270px' }}
        aria-label="Anomaly score distribution histogram"
      >
        <defs>
          <linearGradient id="normalBarGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#94a3b8" stopOpacity="0.7"/>
            <stop offset="100%" stopColor="#475569" stopOpacity="0.45"/>
          </linearGradient>
          <linearGradient id="anomBarGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#f87171" stopOpacity="0.95"/>
            <stop offset="100%" stopColor="#dc2626" stopOpacity="0.9"/>
          </linearGradient>
        </defs>

        {/* Subtle horizontal grid lines */}
        {yTicks.slice(1).map((v, i) => (
          <line key={i}
            x1={PAD.left} y1={baseY - toH(v)}
            x2={PAD.left + cw} y2={baseY - toH(v)}
            stroke="currentColor" strokeOpacity="0.07" strokeWidth="1" strokeDasharray="4,3"
          />
        ))}

        {/* Y axis ticks + labels */}
        {yTicks.map((v, i) => {
          const y = baseY - toH(v)
          return (
            <g key={i}>
              <line x1={PAD.left - 3} y1={y} x2={PAD.left} y2={y}
                stroke="currentColor" strokeOpacity="0.25" strokeWidth="1"/>
              <text x={PAD.left - 6} y={y + 3.5} textAnchor="end"
                fontSize="9" fill="currentColor" fillOpacity="0.4" fontFamily="ui-monospace,monospace">
                {v}
              </text>
            </g>
          )
        })}

        {/* Bars (stacked: normal base + anomalous top) */}
        {bins.map((bin, i) => {
          const x = PAD.left + i * bw + gap / 2
          const barWidth = bw - gap
          const normalH = toH(bin.normal)
          const anomH   = toH(bin.anomalous)
          return (
            <g key={i}>
              {bin.normal > 0 && (
                <rect
                  x={x} y={baseY - normalH}
                  width={barWidth} height={normalH}
                  fill="url(#normalBarGrad)" rx="1"
                />
              )}
              {bin.anomalous > 0 && (
                <rect
                  x={x} y={baseY - normalH - anomH}
                  width={barWidth} height={anomH}
                  fill="url(#anomBarGrad)" rx="1"
                />
              )}
            </g>
          )
        })}

        {/* Decision boundary */}
        {boundaryX && (
          <g>
            <line
              x1={boundaryX} y1={PAD.top - 2}
              x2={boundaryX} y2={baseY + 5}
              stroke="#f97316" strokeWidth="1.5"
              strokeDasharray="4,3" strokeOpacity="0.75"
            />
            <text x={boundaryX + 4} y={PAD.top + 9}
              fontSize="8.5" fill="#f97316" fillOpacity="0.85"
              fontFamily="ui-monospace,monospace">
              0.00
            </text>
          </g>
        )}

        {/* Axes */}
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={baseY}
          stroke="currentColor" strokeOpacity="0.15" strokeWidth="1"/>
        <line x1={PAD.left} y1={baseY} x2={PAD.left + cw} y2={baseY}
          stroke="currentColor" strokeOpacity="0.15" strokeWidth="1"/>

        {/* X axis ticks + labels */}
        {xTicks.map((t, i) => (
          <g key={i}>
            <line x1={t.x} y1={baseY} x2={t.x} y2={baseY + 4}
              stroke="currentColor" strokeOpacity="0.25" strokeWidth="1"/>
            <text x={t.x} y={baseY + 14} textAnchor="middle"
              fontSize="9" fill="currentColor" fillOpacity="0.45"
              fontFamily="ui-monospace,monospace">
              {t.label}
            </text>
          </g>
        ))}

        {/* X axis annotation */}
        <text x={PAD.left + cw / 2} y={H - 4} textAnchor="middle"
          fontSize="8.5" fill="currentColor" fillOpacity="0.3"
          fontFamily="ui-sans-serif,system-ui,sans-serif">
          ← more anomalous · anomaly score · more normal →
        </text>
      </svg>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 pt-2 border-t border-gray-100 dark:border-gray-700/50">
        <div className="text-center">
          <div className="text-sm font-bold text-gray-700 dark:text-gray-300 tabular-nums">
            {scores.length.toLocaleString()}
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">Total sessions</div>
        </div>
        <div className="text-center">
          <div className="text-sm font-bold text-red-500 tabular-nums">
            {nAnomalous.toLocaleString()}
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">Anomalous</div>
        </div>
        <div className="text-center">
          <div className="text-sm font-bold text-gray-700 dark:text-gray-300 tabular-nums">
            {rate}%
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">Anomaly rate</div>
        </div>
      </div>

    </div>
  )
}
