import { useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { apiFetch } from './api/client'
import { Layout } from './components/Layout'
import { AdminPage } from './pages/AdminPage'
import { ChatPage } from './pages/ChatPage'
import { ForgotPasswordPage } from './pages/ForgotPasswordPage'
import { LoginPage } from './pages/LoginPage'
import { RegisterPage } from './pages/RegisterPage'
import { ResetPasswordPage } from './pages/ResetPasswordPage'

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
    <Routes>
      <Route
        element={<Layout user={user} onLogout={handleLogout} />}
      >
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ChatPage user={user} />} />
        <Route path="/chat/:sessionId" element={<ChatPage user={user} />} />
        <Route path="/admin" element={<AdminPage />} />
      </Route>

      <Route path="/login" element={<LoginPage onAuth={setUser} />} />
      <Route path="/register" element={<RegisterPage onAuth={setUser} />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
    </Routes>
  )
}
