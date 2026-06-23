import { Link, useLocation } from 'react-router-dom'

export default function Header({ indexStats }) {
  const { pathname } = useLocation()

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-gray-900 border-b border-gray-700 shrink-0">
      <Link to="/" className="flex items-center gap-3 group">
        <svg className="w-7 h-7 text-blue-400" viewBox="0 0 32 32" fill="none">
          <rect width="32" height="32" rx="6" fill="currentColor" fillOpacity="0.15"/>
          <path d="M6 10h20M6 16h14M6 22h17" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
          <circle cx="25" cy="22" r="4" fill="#ef4444"/>
        </svg>
        <span className="text-lg font-semibold tracking-tight">
          Log<span className="text-blue-400">Sense</span>
        </span>
      </Link>

      <nav className="flex items-center gap-6 text-sm">
        <Link
          to="/"
          className={`transition-colors ${pathname === '/' ? 'text-blue-400' : 'text-gray-400 hover:text-gray-200'}`}
        >
          Upload
        </Link>
        <Link
          to="/dashboard"
          className={`transition-colors ${pathname === '/dashboard' ? 'text-blue-400' : 'text-gray-400 hover:text-gray-200'}`}
        >
          Dashboard
        </Link>
      </nav>

      {indexStats?.size > 0 && (
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span>{indexStats.size} sessions indexed</span>
          <span className="text-gray-600">·</span>
          <span className="text-gray-500">{indexStats.llm_provider || 'offline'}</span>
        </div>
      )}
    </header>
  )
}
