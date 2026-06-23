import { Link, useLocation } from 'react-router-dom'
import { useThemeContext } from '../context/ThemeContext'

function SunIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707M17.657 17.657l-.707-.707M6.343 6.343l-.707-.707M12 8a4 4 0 100 8 4 4 0 000-8z" />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
    </svg>
  )
}

export default function Header({ indexStats }) {
  const { pathname } = useLocation()
  const { dark, toggle } = useThemeContext()

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 shrink-0">
      <Link to="/" className="flex items-center gap-3 group">
        <svg className="w-7 h-7 text-blue-500" viewBox="0 0 32 32" fill="none">
          <rect width="32" height="32" rx="6" fill="currentColor" fillOpacity="0.15"/>
          <path d="M6 10h20M6 16h14M6 22h17" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
          <circle cx="25" cy="22" r="4" fill="#ef4444"/>
        </svg>
        <span className="text-lg font-semibold tracking-tight text-gray-900 dark:text-gray-100">
          Log<span className="text-blue-500">Sense</span>
        </span>
      </Link>

      <nav className="flex items-center gap-6 text-sm">
        <Link to="/"
          className={`transition-colors ${pathname === '/' ? 'text-blue-500' : 'text-gray-500 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'}`}>
          Upload
        </Link>
        <Link to="/dashboard"
          className={`transition-colors ${pathname === '/dashboard' ? 'text-blue-500' : 'text-gray-500 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'}`}>
          Dashboard
        </Link>
      </nav>

      <div className="flex items-center gap-3">
        {indexStats?.size > 0 && (
          <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>{indexStats.size} sessions indexed</span>
            <span className="text-gray-300 dark:text-gray-600">·</span>
            <span>{indexStats.llm_provider || 'offline'}</span>
          </div>
        )}

        {/* Theme toggle */}
        <button
          onClick={toggle}
          title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
          className="p-1.5 rounded-lg text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          {dark ? <SunIcon /> : <MoonIcon />}
        </button>

      </div>
    </header>
  )
}
