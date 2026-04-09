import { Outlet } from 'react-router-dom'
import { ErrorBoundary } from './ErrorBoundary'
import { Header } from './Header'

export function Layout({ user, onLogout }) {
  return (
    <div className="app-shell">
      <Header user={user} onLogout={onLogout} />
      <div className="app-body">
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </div>
    </div>
  )
}
