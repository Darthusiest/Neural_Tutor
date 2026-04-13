import { useEffect, useState } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { apiFetch } from './api/client'
import { Layout } from './components/Layout'
import { ThemeToggle } from './components/ThemeToggle'
import { AdminRoute } from './components/AdminRoute'
import { ProtectedRoute } from './components/ProtectedRoute'
import { AdminPage } from './pages/AdminPage'
import { ChatPage } from './pages/ChatPage'
import { ForgotPasswordPage } from './pages/ForgotPasswordPage'
import { LoginPage } from './pages/LoginPage'
import { RegisterPage } from './pages/RegisterPage'
import { ResetPasswordPage } from './pages/ResetPasswordPage'

const AUTH_PATHS = ['/login', '/register', '/forgot-password', '/reset-password']

export default function App() {
  const [user, setUser] = useState(undefined)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const me = await apiFetch('/api/auth/me')
        if (!cancelled) setUser(me.user)
      } catch {
        if (!cancelled) setUser(null)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  async function handleLogout() {
    try {
      await apiFetch('/api/auth/logout', { method: 'POST' })
    } catch {
      /* ignore */
    }
    setUser(null)
  }

  if (user === undefined) {
    return <div className="center-page muted">Loading…</div>
  }

  return (
    <AppRoutes user={user} onAuth={setUser} onLogout={handleLogout} />
  )
}

function AppRoutes({ user, onAuth, onLogout }) {
  const location = useLocation()
  const showFloatingTheme = AUTH_PATHS.includes(location.pathname)

  return (
    <>
      {showFloatingTheme ? (
        <div className="theme-toggle-floating">
          <ThemeToggle />
        </div>
      ) : null}
      <Routes>
      <Route
        element={<Layout user={user} onLogout={onLogout} />}
      >
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route
          path="/chat"
          element={
            <ProtectedRoute user={user}>
              <ChatPage user={user} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/chat/:sessionId"
          element={
            <ProtectedRoute user={user}>
              <ChatPage user={user} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/admin"
          element={
            <ProtectedRoute user={user}>
              <AdminRoute user={user}>
                <AdminPage />
              </AdminRoute>
            </ProtectedRoute>
          }
        />
      </Route>

      <Route path="/login" element={<LoginPage onAuth={onAuth} />} />
      <Route path="/register" element={<RegisterPage onAuth={onAuth} />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
    </Routes>
    </>
  )
}
