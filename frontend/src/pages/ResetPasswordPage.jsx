import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { apiFetch } from '../api/client'

export function ResetPasswordPage() {
  const [params] = useSearchParams()
  const [token, setToken] = useState(params.get('token') || '')
  const [password, setPassword] = useState('')
  const [msg, setMsg] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    setMsg('')
    try {
      await apiFetch('/api/auth/reset-password', {
        method: 'POST',
        body: JSON.stringify({ token, password }),
      })
      setMsg('Password updated. You can log in.')
    } catch (err) {
      setMsg(err.data?.error || err.message)
    }
  }

  return (
    <div className="center-page">
      <h1>Set new password</h1>
      <form onSubmit={handleSubmit} className="stack-form">
        <input
          value={token}
          onChange={(e) => setToken(e.target.value)}
          placeholder="Reset token"
          required
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="New password"
          required
        />
        {msg ? <p className="muted">{msg}</p> : null}
        <button type="submit">Update password</button>
      </form>
      <p className="muted">
        <Link to="/login">Log in</Link>
      </p>
    </div>
  )
}
