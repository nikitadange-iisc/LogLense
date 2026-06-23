import { useEffect, useRef, useState } from 'react'
import { chat } from '../api/client'

function Message({ role, content }) {
  const isUser = role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`
        max-w-[85%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed
        ${isUser
          ? 'bg-blue-600 text-white rounded-br-none'
          : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-bl-none'
        }
      `}>
        {content}
      </div>
    </div>
  )
}

export default function ChatPanel({ sessions, focusedSessionId, onFocusSession }) {
  const [history, setHistory] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedSession, setSelectedSession] = useState(focusedSessionId || '')
  const bottomRef = useRef(null)

  useEffect(() => {
    if (focusedSessionId) setSelectedSession(focusedSessionId)
  }, [focusedSessionId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history, loading])

  const send = async () => {
    const q = input.trim()
    if (!q || loading) return

    const userMsg = { role: 'user', content: q }
    const newHistory = [...history, userMsg]
    setHistory(newHistory)
    setInput('')
    setLoading(true)

    try {
      const res = await chat(q, selectedSession || null, history)
      setHistory(prev => [...prev, { role: 'assistant', content: res.answer }])
    } catch (err) {
      setHistory(prev => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="card flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 shrink-0 space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-800 dark:text-gray-200">Chat with AI</h2>
          {history.length > 0 && (
            <button
              onClick={() => setHistory([])}
              className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
            >
              Clear
            </button>
          )}
        </div>

        {/* Session focus selector */}
        <select
          value={selectedSession}
          onChange={e => setSelectedSession(e.target.value)}
          className="w-full text-xs bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg px-2.5 py-1.5 text-gray-700 dark:text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="">No session focus (ask about all anomalies)</option>
          {(sessions || []).map(s => (
            <option key={s.session_id} value={s.session_id}>
              {s.session_id} (score: {s.anomaly_score?.toFixed(3)})
            </option>
          ))}
        </select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {history.length === 0 && (
          <div className="text-center text-gray-400 dark:text-gray-600 text-sm mt-8 space-y-3">
            <div className="text-3xl">💬</div>
            <p>Ask anything about the anomalies.</p>
            <div className="space-y-1 text-xs text-gray-400 dark:text-gray-600">
              <p>"What is the most critical anomaly?"</p>
              <p>"Summarise all high-severity issues"</p>
              <p>"What caused the failures in this session?"</p>
            </div>
          </div>
        )}

        {history.map((msg, i) => (
          <Message key={i} role={msg.role} content={msg.content} />
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 dark:bg-gray-700 rounded-xl rounded-bl-none px-4 py-3 flex gap-1">
              {[0, 150, 300].map(d => (
                <span
                  key={d}
                  className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"
                  style={{ animationDelay: `${d}ms` }}
                />
              ))}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-700 shrink-0">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask about the anomalies… (Enter to send)"
            rows={2}
            className="flex-1 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 resize-none focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <button
            onClick={send}
            disabled={!input.trim() || loading}
            className="btn-primary px-3 self-end"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
