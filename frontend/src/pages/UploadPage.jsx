import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Header from '../components/Header'
import FileUpload from '../components/FileUpload'
import PipelineProgress from '../components/PipelineProgress'
import {
  uploadLog, getStatus, getHistory,
  cancelPipeline, resetPipeline, tryout, activateSession, deleteSession,
} from '../api/client'

const DATASETS = [
  { id: 'hdfs',        label: 'HDFS',       desc: 'Hadoop Distributed File System' },
  { id: 'bgl',         label: 'BGL',         desc: 'BlueGene/L supercomputer' },
  { id: 'thunderbird', label: 'Thunderbird', desc: 'Thunderbird supercomputer' },
]

function fmtDate(dt) {
  if (!dt) return ''
  const d = new Date(dt)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

/* ── Session history sidebar ─────────────────────────────────────────────── */
function HistorySidebar({ history, onActivate, onDelete, activatingId }) {
  return (
    <aside className="w-56 shrink-0 flex flex-col gap-1 pr-4 border-r border-gray-200 dark:border-gray-800 overflow-y-auto">
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 shrink-0">
        History
      </p>
      {history.length === 0 ? (
        <p className="text-xs text-gray-500 dark:text-gray-700 leading-relaxed">
          Past analyses will appear here after your first upload.
        </p>
      ) : (
        history.map(item => (
          <div key={item.session_id} className="relative group">
            <button
              onClick={() => onActivate(item.session_id)}
              disabled={activatingId === item.session_id}
              className={`
                w-full text-left rounded-lg px-3 py-2.5 transition-colors border pr-7
                ${activatingId === item.session_id
                  ? 'border-blue-700/50 bg-blue-900/40 opacity-70'
                  : 'border-transparent hover:bg-gray-100 dark:hover:bg-gray-800 hover:border-gray-200 dark:hover:border-gray-700'
                }
              `}
            >
              <div className="flex items-center gap-2 min-w-0">
                <svg className="w-3.5 h-3.5 text-gray-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span className="text-xs text-gray-700 dark:text-gray-300 truncate font-medium">{item.filename}</span>
              </div>
              <p className="mt-0.5 pl-5 text-xs text-gray-500 dark:text-gray-600">
                {item.dataset?.toUpperCase()} · {item.stats?.anomalous_sessions ?? 0} anomalies
              </p>
              <p className="pl-5 text-xs text-gray-500 dark:text-gray-700">{fmtDate(item.created_at)}</p>
            </button>

            {/* Delete button — visible on hover */}
            <button
              onClick={() => onDelete(item.session_id)}
              title="Delete this session"
              className="absolute top-2 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-red-500 dark:hover:text-red-400 p-0.5 rounded"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        ))
      )}
    </aside>
  )
}

/* ── Main component ──────────────────────────────────────────────────────── */
export default function UploadPage() {
  const [status, setStatus]         = useState({ step: 'idle', message: 'Connecting…', progress_pct: 0 })
  const [history, setHistory]       = useState([])
  const [file, setFile]             = useState(null)
  const [dataset, setDataset]       = useState('hdfs')
  const [uploading, setUploading]       = useState(false)
  const [uploadProgress, setUploadProgress] = useState(null)
  const [cancelling, setCancelling] = useState(false)
  const [launching, setLaunching]   = useState(false)
  const [activatingId, setActivatingId] = useState(null)
  const [error, setError]           = useState('')

  const pollRef        = useRef(null)
  // remembers the step on the very first status poll; 'ready' here means
  // we landed on this page after a prior run — don't auto-redirect
  const initialStepRef = useRef(null)
  const navigate       = useNavigate()

  /* ── Polling helpers ──────────────────────────────────────────────────── */
  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }
  const startPolling = () => { stopPolling(); pollRef.current = setInterval(tick, 1500) }

  const refreshHistory = async () => {
    try { setHistory(await getHistory()) } catch {}
  }

  const tick = async () => {
    try {
      const s = await getStatus()
      setStatus(s)

      // Record the step seen on the very first tick
      if (initialStepRef.current === null) {
        initialStepRef.current = s.step
      }

      if (s.step === 'ready' && initialStepRef.current !== 'ready') {
        // Pipeline just completed during this page visit → auto-navigate
        stopPolling()
        await refreshHistory()
        setTimeout(() => navigate('/dashboard'), 600)
        return
      }

      if (s.step === 'error') {
        stopPolling()
        setCancelling(false); setUploading(false); setLaunching(false)
      }
      if (s.step === 'idle') {
        setCancelling(false); setUploading(false); setLaunching(false)
      }
    } catch {
      setStatus(prev => ({ ...prev, message: 'Cannot reach backend — is it running on port 8000?' }))
    }
  }

  useEffect(() => {
    tick()
    startPolling()
    refreshHistory()
    return stopPolling
  }, [])

  /* ── User actions ─────────────────────────────────────────────────────── */
  const handleUpload = async () => {
    if (!file) return
    setError('')
    setUploading(true)
    setUploadProgress(0)
    initialStepRef.current = 'uploading'
    try {
      if (status.step === 'error') await resetPipeline()
      await uploadLog(file, dataset, setUploadProgress)
      setUploadProgress(null)
      startPolling()
    } catch (e) {
      setError(e.message)
      setUploading(false)
      setUploadProgress(null)
    }
  }

  const handleTryout = async () => {
    setError('')
    setLaunching(true)
    initialStepRef.current = 'tryout'
    try {
      await tryout()
      startPolling()
    } catch (e) {
      setError(e.message || 'Demo file (HDFS.log) not found in data/raw/')
      setLaunching(false)
    }
  }

  const handleCancel = async () => {
    setCancelling(true)
    try { await cancelPipeline() } catch { setCancelling(false) }
  }

  const handleRetry = async () => {
    if (!file) return
    setError('')
    try { await resetPipeline() } catch {}
    initialStepRef.current = 'uploading'
    setUploading(true)
    setUploadProgress(0)
    try {
      await uploadLog(file, dataset, setUploadProgress)
      setUploadProgress(null)
      startPolling()
    } catch (e) {
      setError(e.message)
      setUploading(false)
      setUploadProgress(null)
    }
  }

  // Start completely fresh — clear file selection and reset state
  const handleNewAnalysis = async () => {
    try { await resetPipeline() } catch {}
    initialStepRef.current = 'idle'
    setFile(null); setError('')
    setStatus({ step: 'idle', message: 'Ready for new upload.', progress_pct: 0 })
    startPolling()
  }

  const handleActivate = async (sessionId) => {
    setActivatingId(sessionId)
    try {
      await activateSession(sessionId)
      navigate('/dashboard')
    } catch (e) {
      setError(e.message)
      setActivatingId(null)
    }
  }

  const handleDelete = async (sessionId) => {
    try {
      await deleteSession(sessionId)
      setHistory(prev => prev.filter(h => h.session_id !== sessionId))
    } catch (e) {
      setError(e.message)
    }
  }

  /* ── Derived state ────────────────────────────────────────────────────── */
  const isRunning        = !['idle', 'ready', 'error'].includes(status.step)
  const isIdle           = status.step === 'idle'
  const isError          = status.step === 'error'
  // Page was opened while a prior run was already ready
  const isReadyOnLoad    = status.step === 'ready' && initialStepRef.current === 'ready'
  // Show the upload form
  const showForm         = isIdle || isError || isReadyOnLoad
  // Show the pipeline progress widget
  const showProgress     = isRunning || isError

  return (
    <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-950">
      <Header indexStats={null} />

      <main className="flex-1 flex gap-6 p-6 overflow-hidden min-h-0">

        {/* Sidebar */}
        <HistorySidebar
          history={history}
          onActivate={handleActivate}
          onDelete={handleDelete}
          activatingId={activatingId}
        />

        {/* Main content */}
        <div className="flex-1 min-w-0 overflow-y-auto">
          <div className="max-w-xl mx-auto space-y-5">

            {/* Brand */}
            <div className="text-center space-y-1">
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                Log<span className="text-blue-400">Sense</span>
              </h1>
              <p className="text-gray-500 text-sm">RAG-powered log anomaly analysis</p>
            </div>

            {/* "Previous run ready" banner */}
            {isReadyOnLoad && (
              <div className="card p-4 border-green-700/40 bg-green-50 dark:bg-green-950/20 flex items-center justify-between gap-4">
                <div>
                  <p className="text-sm text-green-700 dark:text-green-300 font-medium">Previous analysis ready</p>
                  <p className="text-xs text-green-600 dark:text-green-800 mt-0.5">
                    {status.stats?.anomalous_sessions ?? 0} anomalous sessions indexed
                  </p>
                </div>
                <div className="flex gap-2 shrink-0">
                  <button
                    onClick={() => navigate('/dashboard')}
                    className="btn-primary text-xs py-1.5 px-3"
                  >
                    View Results →
                  </button>
                  <button
                    onClick={handleNewAnalysis}
                    className="btn-secondary text-xs py-1.5 px-3"
                  >
                    New Analysis
                  </button>
                </div>
              </div>
            )}

            {/* Upload form */}
            {showForm && (
              <>
                <FileUpload onFileSelect={setFile} disabled={isRunning || uploading} />

                {/* Dataset selector */}
                <div className="card p-4 space-y-3">
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Dataset type
                  </p>
                  <div className="grid grid-cols-3 gap-2">
                    {DATASETS.map(ds => (
                      <button
                        key={ds.id}
                        onClick={() => setDataset(ds.id)}
                        className={`
                          rounded-lg p-3 text-left border transition-colors
                          ${dataset === ds.id
                            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300'
                            : 'border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 text-gray-600 dark:text-gray-400 hover:border-gray-400 dark:hover:border-gray-600'
                          }
                        `}
                      >
                        <p className="text-sm font-semibold">{ds.label}</p>
                        <p className="text-xs opacity-60 mt-0.5 leading-tight">{ds.desc}</p>
                      </button>
                    ))}
                  </div>
                </div>

                {error && (
                  <p className="text-xs text-red-400 bg-red-950/30 border border-red-800/40 rounded-lg px-3 py-2">
                    {error}
                  </p>
                )}

                <div className="flex gap-3">
                  <button
                    onClick={handleUpload}
                    disabled={!file || uploading}
                    className="btn-primary flex-1 justify-center"
                  >
                    {uploading && uploadProgress !== null && uploadProgress < 100
                      ? `Uploading… ${uploadProgress}%`
                      : uploading
                        ? 'Starting pipeline…'
                        : 'Upload & Analyze'}
                  </button>

                  <button
                    onClick={handleTryout}
                    disabled={launching}
                    className="btn-secondary flex items-center gap-2"
                    title="Runs on first 5 000 lines of HDFS.log (requires data/raw/HDFS.log)"
                  >
                    {launching
                      ? <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/></svg>
                      : <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                    }
                    Quick Demo
                  </button>
                </div>
              </>
            )}

            {/* File upload progress bar */}
            {uploading && uploadProgress !== null && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                  <span>{uploadProgress < 100 ? 'Uploading file to server…' : 'Upload complete, starting pipeline…'}</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-blue-500 h-1.5 rounded-full transition-all duration-200"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Pipeline progress widget */}
            {showProgress && <PipelineProgress status={status} />}

            {/* Cancel button */}
            {isRunning && (
              <button
                onClick={handleCancel}
                disabled={cancelling}
                className="btn-secondary w-full justify-center flex items-center gap-2 border-red-800/50 text-red-400 hover:bg-red-950/30 disabled:opacity-50"
              >
                {cancelling
                  ? <><svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/></svg>Cancelling…</>
                  : <><svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12"/></svg>Cancel pipeline</>
                }
              </button>
            )}

            {/* Error actions */}
            {isError && (
              <div className="flex gap-3">
                {file && (
                  <button
                    onClick={handleRetry}
                    disabled={uploading}
                    className="btn-primary flex-1 justify-center"
                  >
                    {uploading ? 'Starting…' : '↺ Retry same file'}
                  </button>
                )}
                <button
                  onClick={handleNewAnalysis}
                  className="btn-secondary flex-1 justify-center"
                >
                  New Analysis
                </button>
              </div>
            )}

          </div>
        </div>
      </main>
    </div>
  )
}
