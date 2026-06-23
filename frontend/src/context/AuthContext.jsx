import { createContext, useContext, useEffect, useState } from 'react'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser]   = useState(null)   // username string or null
  const [ready, setReady] = useState(false)  // true once initial token check is done

  // Verify stored token on mount
  useEffect(() => {
    const token = localStorage.getItem('token')
    if (!token) { setReady(true); return }

    fetch('/api/auth/me', {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => setUser(data.username))
      .catch(() => localStorage.removeItem('token'))
      .finally(() => setReady(true))
  }, [])

  const login = (token, username) => {
    localStorage.setItem('token', token)
    setUser(username)
  }

  const logout = () => {
    localStorage.removeItem('token')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, ready, login, logout, isAuth: !!user }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
