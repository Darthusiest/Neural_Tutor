import { Link } from 'react-router-dom'

export function Header({ user, onLogout }) {
  return (
    <header className="app-header">
      <Link className="brand" to="/chat">LING 487 Tutor</Link>
      <nav className="header-nav">
        {user ? (
          <>
            {user.is_admin ? <Link to="/admin">Admin</Link> : null}
            <span className="muted">{user.email}</span>
            <button type="button" className="link-btn" onClick={onLogout}>
              Log out
            </button>
          </>
        ) : (
          <>
            <Link to="/login">Log in</Link>
            <Link to="/register">Sign up</Link>
          </>
        )}
      </nav>
    </header>
  )
}
