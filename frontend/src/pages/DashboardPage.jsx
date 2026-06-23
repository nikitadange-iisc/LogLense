import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Header from '../components/Header'
import AnomalyTable from '../components/AnomalyTable'
import SessionDrawer from '../components/SessionDrawer'
import ChatPanel from '../components/ChatPanel'
import { getSessions, getStatus, getHistory, activateSession, resetPipeline } from '../api/client'

function fmt(dt) {
  if (!dt) return ''
  const d = new Date(dt)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

/* ── Session history panel ───────────────────────────────────────────────── */
function HistoryPanel({ history, activeId, onActivate, onNewAnalysis }) {
  return (
    <div className="w-52 shrink-0 flex flex-col border-r border-gray-800 overflow-hidden">
      <div className="px-3 py-3 border-b border-gray-800 flex items-center justify-between">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Sessions</span>
        <button
          onClick={onNewAnalysis}
          className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
          title="Upload a new log file"
        >
          + New
        </button>
      </div>

      <div className="flex-1 overflow-y-auto py-1">
        {history.length === 0 ? (
          <p className="text-xs text-gray-700 px-3 py-2">No past sessions.</p>
        ) : (
          history.map(item => (
            <button
              key={item.session_id}
              onClick={() => onActivate(item.session_id)}
              className={`
                w-full text-left px-3 py-2.5 transition-colors
                ${activeId === item.session_id
                  ? 'bg-blue-900/30 border-l-2 border-blue-500'
                  : 'border-l-2 border-transparent hover:bg-gray-800/60'
                }
              `}
            >
              <p className="text-xs text-gray-300 font-medium truncate">{item.filename}</p>
              <p className="text-xs text-gray-600 mt-0.5">
                {item.stats?.anomalous_sessions ?? 0} anomalies · {item.dataset?.toUpperCase()}
              </p>
              <p className="text-xs text-gray-700">{fmt(item.created_at)}</p>
            </button>
          ))
        )}
      </div>
    </div>
  )
}

/* ── Main component ──────────────────────────────────────────────────────── */
export default function DashboardPage() {
  const [sessions, setSessions]         = useState([])
  const [indexStats, setIndexStats]     = useState(null)
  const [drawerSessionId, setDrawerSessionId] = useState(null)
  const [chatFocusId, setChatFocusId]   = useState(null)
  const [loading, setLoading]           = useState(true)
  const [history, setHistory]           = useState([])
  const [activeHistoryId, setActiveHistoryId] = useState(null)
  const [activating, setActivating]     = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    const load = async () => {
      try {
        const [status, data, hist] = await Promise.all([
          getStatus(), getSessions(), getHistory(),
        ])
        if (status.step === 'idle') { navigate('/'); return }
        setIndexStats(status.stats)
        setSessions(data)
        setHistory(hist)
        if (status.step === 'ready') {
          // Find which history entry matches current active session
          const activeId = hist[0]?.session_id ?? null
          setActiveHistoryId(activeId)
        }
      } catch (e) {
        console.error('Failed to load dashboard:', e)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const handleAnalysisComplete = (sessionId, result) => {
    setSessions(prev =>
      prev.map(s =>
        s.session_id === sessionId
          ? { ...s, severity: result.severity, analyzed: true }
          : s
      )
    )
  }

  const handleSessionClick = (sessionId) => {
    setDrawerSessionId(sessionId)
    setChatFocusId(sessionId)
  }

  const handleActivateHistory = async (sessionId) => {
    if (sessionId === activeHistoryId) return
    setActivating(true)
    try {
      await activateSession(sessionId)
      const [status, data] = await Promise.all([getStatus(), getSessions()])
      setIndexStats(status.stats)
      setSessions(data)
      setActiveHistoryId(sessionId)
      setDrawerSessionId(null)
      setChatFocusId(null)
    } catch (e) {
      console.error('Failed to switch session:', e)
    } finally {
      setActivating(false)
    }
  }

  const handleNewAnalysis = async () => {
    try { await resetPipeline() } catch {}
    navigate('/')
  }

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col bg-gray-950">
        <Header indexStats={null} />
        <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
          Loading…
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-950">
      <Header indexStats={indexStats} />

      <main className="flex-1 flex overflow-hidden min-h-0">

        {/* Session history sidebar */}
        <HistoryPanel
          history={history}
          activeId={activeHistoryId}
          onActivate={handleActivateHistory}
          onNewAnalysis={handleNewAnalysis}
        />

        {/* Anomaly table + chat */}
        <div className={`flex-1 flex gap-4 p-4 overflow-hidden min-h-0 transition-opacity ${activating ? 'opacity-40 pointer-events-none' : ''}`}>
          <div className="flex-[3] min-w-0 overflow-hidden">
            <AnomalyTable
              sessions={sessions}
              onSessionClick={handleSessionClick}
              onAnalysisComplete={handleAnalysisComplete}
            />
          </div>
          <div className="flex-[2] min-w-0 overflow-hidden">
            <ChatPanel
              sessions={sessions}
              focusedSessionId={chatFocusId}
              onFocusSession={setChatFocusId}
            />
          </div>
        </div>

      </main>

      {drawerSessionId && (
        <SessionDrawer
          sessionId={drawerSessionId}
          onClose={() => setDrawerSessionId(null)}
          onAnalysisComplete={handleAnalysisComplete}
        />
      )}
    </div>
  )
}
