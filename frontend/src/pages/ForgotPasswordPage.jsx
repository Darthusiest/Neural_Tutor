import { useState } from 'react'
import { Link } from 'react-router-dom'
import { apiFetch } from '../api/client'

export function ForgotPasswordPage() {
  const [msg, setMsg] = useState('')
  const [devToken, setDevToken] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    setMsg('')
    setDevToken('')
    const fd = new FormData(e.target)
    try {
      const data = await apiFetch('/api/auth/forgot-password', {
        method: 'POST',
        body: JSON.stringify({ email: fd.get('email') }),
      })
      setMsg(data.message || 'Check your email.')
      if (data.dev_reset_token) setDevToken(data.dev_reset_token)
    } catch (err) {
      setMsg(err.data?.error || err.message)
    }
  }

  return (
    <div className="center-page">
      <h1>Reset password</h1>
      <form onSubmit={handleSubmit} className="stack-form">
        <input name="email" type="email" placeholder="Email" required />
        {msg ? <p className="muted">{msg}</p> : null}
        {devToken ? (
          <p className="muted">
            Dev only: use this token on{' '}
            <Link to="/reset-password">reset page</Link>:{' '}
            <code>{devToken}</code>
          </p>
        ) : null}
        <button type="submit">Send reset link</button>
      </form>
      <p className="muted">
        <Link to="/login">Back to login</Link>
      </p>
    </div>
  )
}
