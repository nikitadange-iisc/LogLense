import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Header from '../components/Header'
import AnomalyTable from '../components/AnomalyTable'
import SessionDrawer from '../components/SessionDrawer'
import ChatPanel from '../components/ChatPanel'
import LogViewer from '../components/LogViewer'
import { getSessions, getStatus, getHistory, activateSession, resetPipeline, deleteSession, renameSession, getScores } from '../api/client'
import ScoreDistributionChart from '../components/ScoreDistributionChart'

function fmt(dt) {
  if (!dt) return ''
  const d = new Date(dt)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

/* ── Session history panel ───────────────────────────────────────────────── */
function HistoryPanel({ history, activeId, onActivate, onDelete, onRename, onNewAnalysis }) {
  const [editingId, setEditingId] = useState(null)
  const [editValue, setEditValue]  = useState('')
  const inputRef = useRef(null)

  const startEdit = (e, item) => {
    e.stopPropagation()
    setEditingId(item.session_id)
    setEditValue(item.filename)
    setTimeout(() => inputRef.current?.select(), 0)
  }

  const commitEdit = (sessionId) => {
    const name = editValue.trim()
    if (name) onRename(sessionId, name)
    setEditingId(null)
  }

  const onKeyDown = (e, sessionId) => {
    if (e.key === 'Enter')  { e.preventDefault(); commitEdit(sessionId) }
    if (e.key === 'Escape') setEditingId(null)
  }

  return (
    <div className="w-52 shrink-0 flex flex-col border-r border-gray-200 dark:border-gray-800 overflow-hidden">
      <div className="px-3 py-3 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Sessions</span>
        <button
          onClick={onNewAnalysis}
          className="text-xs text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 transition-colors"
          title="Upload a new log file"
        >
          + New
        </button>
      </div>

      <div className="flex-1 overflow-y-auto py-1">
        {history.length === 0 ? (
          <p className="text-xs text-gray-500 px-3 py-2">No past sessions.</p>
        ) : (
          history.map(item => {
            const isEditing = editingId === item.session_id
            const rowCls = `w-full text-left px-3 py-2.5 transition-colors pr-14 ${
              activeId === item.session_id
                ? 'bg-blue-50 dark:bg-blue-900/30 border-l-2 border-blue-500'
                : 'border-l-2 border-transparent hover:bg-gray-100 dark:hover:bg-gray-800/60'
            }`
            return (
              <div key={item.session_id} className="relative group">
                {isEditing ? (
                  <div className={rowCls}>
                    <input
                      ref={inputRef}
                      value={editValue}
                      onChange={e => setEditValue(e.target.value)}
                      onBlur={() => commitEdit(item.session_id)}
                      onKeyDown={e => onKeyDown(e, item.session_id)}
                      className="w-full text-xs bg-white dark:bg-gray-900 border border-blue-400 rounded px-1 py-0.5 text-gray-700 dark:text-gray-200 outline-none"
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-600 mt-0.5">
                      {item.stats?.anomalous_sessions ?? 0} anomalies · {item.dataset?.toUpperCase()}
                    </p>
                    <p className="text-xs text-gray-400 dark:text-gray-700">{fmt(item.created_at)}</p>
                  </div>
                ) : (
                  <button
                    onClick={() => onActivate(item.session_id)}
                    className={rowCls}
                  >
                    <p className="text-xs text-gray-700 dark:text-gray-300 font-medium truncate">{item.filename}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-600 mt-0.5">
                      {item.stats?.anomalous_sessions ?? 0} anomalies · {item.dataset?.toUpperCase()}
                    </p>
                    <p className="text-xs text-gray-400 dark:text-gray-700">{fmt(item.created_at)}</p>
                  </button>
                )}

                {/* Rename + Delete — visible on hover */}
                {!isEditing && (
                  <div className="absolute top-2 right-1 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={e => startEdit(e, item)}
                      title="Rename this session"
                      className="text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 p-0.5 rounded"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                      </svg>
                    </button>
                    <button
                      onClick={() => onDelete(item.session_id)}
                      title="Delete this session"
                      className="text-gray-400 hover:text-red-500 dark:hover:text-red-400 p-0.5 rounded"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                )}
              </div>
            )
          })
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
  const [scores, setScores]             = useState([])
  const [activeTab, setActiveTab]       = useState(
    () => localStorage.getItem('dashboard_tab') || 'anomalies'
  )
  const navigate = useNavigate()

  useEffect(() => { localStorage.setItem('dashboard_tab', activeTab) }, [activeTab])
  useEffect(() => {
    if (activeHistoryId) localStorage.setItem('dashboard_session', activeHistoryId)
  }, [activeHistoryId])

  useEffect(() => {
    const load = async () => {
      try {
        const [status, data, hist, scoreData] = await Promise.all([
          getStatus(), getSessions(), getHistory(), getScores(),
        ])
        setScores(scoreData || [])
        if (status.step === 'idle') { navigate('/'); return }
        setIndexStats(status.stats)
        setSessions(data)
        setHistory(hist)
        // Use the backend's authoritative active_session_id first,
        // fall back to localStorage, then the newest history entry.
        const backendActiveId = status.active_session_id || ''
        const storedId = localStorage.getItem('dashboard_session') || ''
        const inHist = (id) => id && hist.find(h => h.session_id === id)?.session_id
        const activeId = inHist(backendActiveId) || inHist(storedId) || hist[0]?.session_id || null
        setActiveHistoryId(activeId)
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
      const [status, data, scoreData] = await Promise.all([getStatus(), getSessions(), getScores()])
      setIndexStats(status.stats)
      setSessions(data)
      setScores(scoreData || [])
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

  const handleRenameHistory = async (sessionId, newName) => {
    try {
      await renameSession(sessionId, newName)
      setHistory(prev => prev.map(h =>
        h.session_id === sessionId ? { ...h, filename: newName } : h
      ))
    } catch (e) {
      console.error('Failed to rename session:', e)
    }
  }

  const handleDeleteHistory = async (sessionId) => {
    try {
      await deleteSession(sessionId)
      setHistory(prev => prev.filter(h => h.session_id !== sessionId))
      // If the deleted session was active, clear the dashboard
      if (sessionId === activeHistoryId) {
        setActiveHistoryId(null)
        setSessions([])
        setIndexStats(null)
        setDrawerSessionId(null)
        setChatFocusId(null)
        localStorage.removeItem('dashboard_session')
      }
    } catch (e) {
      console.error('Failed to delete session:', e)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-950">
        <Header indexStats={null} />
        <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
          Loading…
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-950">
      <Header indexStats={indexStats} />

      <main className="flex-1 flex overflow-hidden min-h-0">

        {/* Session history sidebar */}
        <HistoryPanel
          history={history}
          activeId={activeHistoryId}
          onActivate={handleActivateHistory}
          onDelete={handleDeleteHistory}
          onRename={handleRenameHistory}
          onNewAnalysis={handleNewAnalysis}
        />

        {/* Main content + chat */}
        <div className={`flex-1 flex gap-4 p-4 overflow-hidden min-h-0 transition-opacity ${activating ? 'opacity-40 pointer-events-none' : ''}`}>

          {/* Left panel: tab bar + tabbed content */}
          <div className="flex-[3] min-w-0 flex flex-col overflow-hidden">

            {/* Tab bar */}
            <div className="flex items-center gap-1 mb-3 shrink-0">
              {[
                { id: 'anomalies',    label: 'Anomalies',    count: sessions.length },
                { id: 'distribution', label: 'Distribution' },
                { id: 'logs',         label: 'Log Viewer' },
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors
                    ${activeTab === tab.id
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-800'
                    }
                  `}
                >
                  {tab.label}
                  {tab.count != null && (
                    <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold
                      ${activeTab === tab.id ? 'bg-blue-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'}`}>
                      {tab.count}
                    </span>
                  )}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div className="flex-1 min-h-0 overflow-hidden overflow-y-auto">
              {activeTab === 'anomalies' ? (
                <AnomalyTable
                  sessions={sessions}
                  onSessionClick={handleSessionClick}
                  onAnalysisComplete={handleAnalysisComplete}
                />
              ) : activeTab === 'distribution' ? (
                <ScoreDistributionChart scores={scores} />
              ) : (
                <LogViewer sessionKey={activeHistoryId} />
              )}
            </div>
          </div>

          {/* Right panel: chat (anomalies tab only) */}
          {activeTab === 'anomalies' && (
            <div className="flex-[2] min-w-0 overflow-hidden">
              <ChatPanel
                sessions={sessions}
                focusedSessionId={chatFocusId}
                onFocusSession={setChatFocusId}
              />
            </div>
          )}
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
