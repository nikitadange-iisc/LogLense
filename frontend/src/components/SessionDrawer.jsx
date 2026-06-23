import { useEffect, useState } from 'react'
import { getSession, analyzeSession } from '../api/client'

const SEV_CLASS = {
  critical: 'severity-critical',
  high:     'severity-high',
  medium:   'severity-medium',
  low:      'severity-low',
}

function ConfidenceBar({ value }) {
  const pct = Math.round((value || 0) * 100)
  const color = pct >= 80 ? 'bg-green-500' : pct >= 50 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-400 w-8 text-right">{pct}%</span>
    </div>
  )
}

export default function SessionDrawer({ sessionId, onClose, onAnalysisComplete }) {
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!sessionId) return
    setLoading(true)
    setError(null)
    getSession(sessionId)
      .then(setSession)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [sessionId])

  const handleAnalyze = async () => {
    setAnalyzing(true)
    try {
      const result = await analyzeSession(sessionId)
      setSession(s => ({ ...s, analysis: result }))
      onAnalysisComplete(sessionId, result)
    } catch (e) {
      setError(e.message)
    } finally {
      setAnalyzing(false)
    }
  }

  const analysis = session?.analysis

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-30"
        onClick={onClose}
      />

      {/* Drawer */}
      <aside className="fixed right-0 top-0 h-full w-[520px] max-w-full bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-700 z-40 flex flex-col overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between p-4 border-b border-gray-200 dark:border-gray-700 shrink-0">
          <div className="flex-1 min-w-0">
            <p className="text-xs text-gray-500 mb-0.5">Session</p>
            <p className="font-mono text-sm text-blue-600 dark:text-blue-300 truncate">{sessionId}</p>
          </div>
          <button onClick={onClose} className="ml-3 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center h-40 text-gray-500 text-sm">
              Loading session…
            </div>
          )}
          {error && (
            <div className="m-4 p-3 rounded-lg bg-red-900/30 text-red-400 text-sm">{error}</div>
          )}

          {session && !loading && (
            <div className="p-4 space-y-4">
              {/* Meta */}
              <div className="grid grid-cols-3 gap-3">
                <div className="card p-3">
                  <p className="text-xs text-gray-500">Anomaly Score</p>
                  <p className="font-mono text-sm text-orange-400 mt-0.5">
                    {session.anomaly_score?.toFixed(4) ?? '—'}
                  </p>
                </div>
                <div className="card p-3">
                  <p className="text-xs text-gray-500">Lines</p>
                  <p className="text-sm text-gray-200 mt-0.5">{session.raw_lines?.length ?? 0}</p>
                </div>
                <div className="card p-3">
                  <p className="text-xs text-gray-500">Range</p>
                  <p className="font-mono text-xs text-gray-200 mt-0.5">
                    {session.line_range ? `${session.line_range[0]}–${session.line_range[1]}` : '—'}
                  </p>
                </div>
              </div>

              {/* Raw log lines */}
              <div>
                <p className="text-xs text-gray-500 mb-2">Raw Log Lines</p>
                <pre className="bg-gray-950 rounded-lg p-3 text-xs font-mono text-green-300 overflow-x-auto max-h-52 whitespace-pre-wrap leading-5">
                  {(session.raw_lines || []).join('\n') || '(no lines)'}
                </pre>
              </div>

              {/* Analysis */}
              {analysis ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">AI Analysis</h3>
                    {analysis.severity && (
                      <span className={`text-xs px-2 py-0.5 rounded ${SEV_CLASS[analysis.severity] || 'severity-unknown'}`}>
                        {analysis.severity}
                      </span>
                    )}
                  </div>

                  {analysis.root_cause && (
                    <div className="card p-3">
                      <p className="text-xs text-gray-500 mb-1">Root Cause</p>
                      <p className="text-sm text-gray-100">{analysis.root_cause}</p>
                    </div>
                  )}

                  {analysis.confidence != null && (
                    <div>
                      <p className="text-xs text-gray-500 mb-1.5">Confidence</p>
                      <ConfidenceBar value={analysis.confidence} />
                    </div>
                  )}

                  {analysis.explanation && (
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Explanation</p>
                      <p className="text-sm text-gray-300 leading-relaxed">{analysis.explanation}</p>
                    </div>
                  )}

                  {analysis.failure_trace?.length > 0 && (
                    <div>
                      <p className="text-xs text-gray-500 mb-1.5">Failure Trace</p>
                      <div className="space-y-1.5">
                        {analysis.failure_trace.map((t, i) => (
                          <div key={i} className="bg-gray-100 dark:bg-gray-800 rounded p-2">
                            <p className="font-mono text-xs text-green-700 dark:text-green-300 mb-0.5 break-all">{t.line}</p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">{t.annotation}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {analysis.recommended_action && (
                    <div className="card p-3 border-blue-800/50 bg-blue-950/20">
                      <p className="text-xs text-blue-400 mb-1">Recommended Action</p>
                      <p className="text-sm text-gray-200">{analysis.recommended_action}</p>
                    </div>
                  )}

                  {analysis.note && (
                    <p className="text-xs text-gray-500 italic">{analysis.note}</p>
                  )}
                </div>
              ) : (
                <button
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className="btn-primary w-full justify-center"
                >
                  {analyzing ? (
                    <>
                      <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Analysing with AI…
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                      Analyse with AI
                    </>
                  )}
                </button>
              )}
            </div>
          )}
        </div>
      </aside>
    </>
  )
}
