import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { apiFetch } from '../api/client'

export function ResetPasswordPage() {
  const [params] = useSearchParams()
  const token = params.get('token') || ''
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [msg, setMsg] = useState('')
  const [success, setSuccess] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setMsg('')

    if (!token) {
      setMsg('Missing reset token. Use the link in your email.')
      return
    }
    if (password !== confirmPassword) {
      setMsg('Passwords do not match.')
      return
    }

    try {
      await apiFetch('/api/auth/reset-password', {
        method: 'POST',
        body: JSON.stringify({ token, password }),
      })
      setSuccess(true)
      setMsg('Password updated. You can log in.')
    } catch (err) {
      setMsg(err.data?.error || err.message)
    }
  }

  return (
    <div className="center-page">
      <h1>Set new password</h1>
      {!token ? (
        <p className="muted">
          This page requires a reset token from the email link. Return to{' '}
          <Link to="/forgot-password">forgot password</Link> to request a new one.
        </p>
      ) : (
        <form onSubmit={handleSubmit} className="stack-form">
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="New password"
            autoComplete="new-password"
            required
          />
          <input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="Confirm new password"
            autoComplete="new-password"
            required
          />
          {msg ? <p className="muted">{msg}</p> : null}
          <button type="submit" disabled={success}>
            Update password
          </button>
        </form>
      )}
      <p className="muted">
        <Link to="/login">Log in</Link>
      </p>
    </div>
  )
}
