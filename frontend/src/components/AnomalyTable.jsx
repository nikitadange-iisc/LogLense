import { useState } from 'react'
import { analyzeSession } from '../api/client'

const SEV_CLASS = {
  critical: 'severity-critical',
  high:     'severity-high',
  medium:   'severity-medium',
  low:      'severity-low',
}

function SeverityBadge({ severity }) {
  const cls = SEV_CLASS[severity] || 'severity-unknown'
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {severity || 'pending'}
    </span>
  )
}

export default function AnomalyTable({ sessions, onSessionClick, onAnalysisComplete }) {
  const [analyzing, setAnalyzing] = useState(null)

  const handleAnalyze = async (e, sessionId) => {
    e.stopPropagation()
    setAnalyzing(sessionId)
    try {
      const result = await analyzeSession(sessionId)
      onAnalysisComplete(sessionId, result)
    } catch (err) {
      alert(`Analysis failed: ${err.message}`)
    } finally {
      setAnalyzing(null)
    }
  }

  if (!sessions?.length) {
    return (
      <div className="card p-8 text-center text-gray-500">
        No anomalous sessions found. Upload a log file to get started.
      </div>
    )
  }

  return (
    <div className="card overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
          Anomalous Sessions
          <span className="ml-2 text-xs text-gray-500 font-normal">{sessions.length} total</span>
        </h2>
        <span className="text-xs text-gray-500">sorted by anomaly score</span>
      </div>

      <div className="overflow-y-auto max-h-[calc(100vh-240px)]">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
            <tr>
              <th className="text-left px-4 py-2 text-xs text-gray-500 dark:text-gray-400 font-medium">Session ID</th>
              <th className="text-left px-4 py-2 text-xs text-gray-500 dark:text-gray-400 font-medium">Severity</th>
              <th className="text-right px-4 py-2 text-xs text-gray-500 dark:text-gray-400 font-medium">Score</th>
              <th className="text-right px-4 py-2 text-xs text-gray-500 dark:text-gray-400 font-medium">Lines</th>
              <th className="px-4 py-2" />
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <tr
                key={s.session_id}
                onClick={() => onSessionClick(s.session_id)}
                className="border-b border-gray-200/70 dark:border-gray-700/50 hover:bg-gray-50 dark:hover:bg-gray-700/40 cursor-pointer transition-colors"
              >
                <td className="px-4 py-2.5 font-mono text-xs text-blue-600 dark:text-blue-300 max-w-[200px] truncate">
                  {s.session_id}
                </td>
                <td className="px-4 py-2.5">
                  <SeverityBadge severity={s.severity} />
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-700 dark:text-gray-300">
                  {s.anomaly_score?.toFixed(4) ?? '—'}
                </td>
                <td className="px-4 py-2.5 text-right text-xs text-gray-500 dark:text-gray-400">
                  {s.num_lines}
                </td>
                <td className="px-4 py-2.5 text-right">
                  {s.analyzed ? (
                    <span className="text-xs text-green-600 dark:text-green-500">✓ done</span>
                  ) : (
                    <button
                      onClick={(e) => handleAnalyze(e, s.session_id)}
                      disabled={analyzing === s.session_id}
                      className="text-xs px-2 py-1 rounded bg-blue-100 text-blue-600 hover:bg-blue-200 dark:bg-blue-900/50 dark:text-blue-300 dark:hover:bg-blue-800/60 transition-colors disabled:opacity-50"
                    >
                      {analyzing === s.session_id ? '...' : 'Analyze'}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
