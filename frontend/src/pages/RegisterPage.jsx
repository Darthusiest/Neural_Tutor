import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { apiFetch } from '../api/client'

export function RegisterPage({ onAuth }) {
  const nav = useNavigate()
  const [error, setError] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    const fd = new FormData(e.target)
    try {
      await apiFetch('/api/auth/register', {
        method: 'POST',
        body: JSON.stringify({
          email: fd.get('email'),
          password: fd.get('password'),
        }),
      })
      const me = await apiFetch('/api/auth/me')
      onAuth(me.user)
      nav('/chat')
    } catch (err) {
      setError(err.data?.error || err.message)
    }
  }

  return (
    <div className="center-page">
      <h1>Sign up</h1>
      <form onSubmit={handleSubmit} className="stack-form">
        <input name="email" type="email" placeholder="Email" required />
        <input name="password" type="password" placeholder="Password" required />
        {error ? <p className="error">{error}</p> : null}
        <button type="submit">Create account</button>
      </form>
      <p className="muted">
        <Link to="/login">Already have an account?</Link>
      </p>
    </div>
  )
}
