import { useEffect, useState } from 'react'
import { getLogs } from '../api/client'

const PER_PAGE = 200

/* Build the list of page indices to show in the paginator.
   Always includes page 0 and lastPage; shows a window of ±2 around current;
   inserts null as an ellipsis placeholder between gaps. */
function buildPageItems(current, totalPages) {
  if (totalPages <= 1) return []
  const last = totalPages - 1
  const set  = new Set([0, last])
  for (let i = Math.max(0, current - 2); i <= Math.min(last, current + 2); i++) set.add(i)
  const sorted = [...set].sort((a, b) => a - b)
  const items  = []
  for (let i = 0; i < sorted.length; i++) {
    if (i > 0 && sorted[i] - sorted[i - 1] > 1) items.push(null)
    items.push(sorted[i])
  }
  return items
}

export default function LogViewer({ sessionKey }) {
  const [lines,      setLines]      = useState([])
  const [page,       setPage]       = useState(0)
  const [hasNext,    setHasNext]    = useState(false)
  const [totalLines, setTotalLines] = useState(0)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)

  const totalPages = totalLines > 0 ? Math.ceil(totalLines / PER_PAGE) : 0

  useEffect(() => {
    goToPage(0)
  }, [sessionKey]) // eslint-disable-line react-hooks/exhaustive-deps

  async function goToPage(p) {
    setLoading(true)
    setError(null)
    try {
      const data = await getLogs(p)
      setLines(data.lines)
      setPage(p)
      setHasNext(data.has_next)
      if (data.total_lines > 0) setTotalLines(data.total_lines)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const lineStart = page * PER_PAGE + 1
  const lineEnd   = totalLines > 0
    ? Math.min((page + 1) * PER_PAGE, totalLines)
    : (page + 1) * PER_PAGE

  const pageItems = buildPageItems(page, totalPages)

  return (
    <div className="h-full flex flex-col bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">

      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200 dark:border-gray-800 shrink-0">
        <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Raw Logs</span>
        {lines.length > 0 && (
          <span className="text-xs text-gray-400 dark:text-gray-600">
            Lines {lineStart.toLocaleString()}–{lineEnd.toLocaleString()}
            {totalLines > 0 && ` of ${totalLines.toLocaleString()}`}
          </span>
        )}
      </div>

      {/* Log lines */}
      <div className="flex-1 overflow-y-auto font-mono text-[11px] leading-5">
        {error ? (
          <div className="p-4 text-red-500 text-xs">{error}</div>
        ) : lines.length === 0 && !loading ? (
          <div className="p-4 text-gray-400 dark:text-gray-600 text-xs text-center">
            No log lines available.
          </div>
        ) : (
          lines.map((line, i) => (
            <div
              key={`${line.line_number}-${i}`}
              className="flex items-start gap-2 px-3 py-px hover:bg-gray-50 dark:hover:bg-gray-800/40 border-b border-gray-100/50 dark:border-gray-800/20"
            >
              <span className="text-gray-300 dark:text-gray-700 w-9 text-right shrink-0 select-none pt-px">
                {line.line_number}
              </span>
              <span className="text-gray-700 dark:text-gray-300 break-all flex-1">
                {line.content}
              </span>
            </div>
          ))
        )}

        {loading && (
          <div className="flex items-center justify-center h-24 text-gray-400 dark:text-gray-600 text-xs gap-2">
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
            </svg>
            Loading…
          </div>
        )}
      </div>

      {/* Paginator */}
      {(totalPages > 1 || page > 0 || hasNext) && (
        <div className="shrink-0 flex items-center justify-center gap-1 px-3 py-2.5 border-t border-gray-200 dark:border-gray-800 flex-wrap">

          {/* Prev */}
          <button
            onClick={() => goToPage(page - 1)}
            disabled={page === 0 || loading}
            className="px-2.5 py-1 text-xs rounded border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            ← Prev
          </button>

          {/* Page numbers */}
          {pageItems.map((p, i) =>
            p === null ? (
              <span key={`ellipsis-${i}`} className="text-xs text-gray-400 dark:text-gray-600 px-1 select-none">
                …
              </span>
            ) : (
              <button
                key={p}
                onClick={() => p !== page && goToPage(p)}
                disabled={loading}
                title={`Lines ${(p * PER_PAGE + 1).toLocaleString()}–${Math.min((p + 1) * PER_PAGE, totalLines || (p + 1) * PER_PAGE).toLocaleString()}`}
                className={`min-w-[2rem] px-2 py-1 text-xs rounded border transition-colors ${
                  p === page
                    ? 'bg-blue-600 border-blue-600 text-white font-semibold cursor-default'
                    : 'border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-40'
                }`}
              >
                {p + 1}
              </button>
            )
          )}

          {/* Next */}
          <button
            onClick={() => goToPage(page + 1)}
            disabled={!hasNext || loading}
            className="px-2.5 py-1 text-xs rounded border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}
