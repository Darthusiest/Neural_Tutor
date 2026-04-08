import { Outlet } from 'react-router-dom'
import { Header } from './Header'

export function Layout({ user, onLogout }) {
  return (
    <div className="app-shell">
      <Header user={user} onLogout={onLogout} />
      <div className="app-body">
        <Outlet />
      </div>
    </div>
  )
}
