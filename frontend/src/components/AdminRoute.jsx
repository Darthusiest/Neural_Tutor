import { Navigate } from 'react-router-dom'

/** Requires logged-in admin (`user.is_admin`). Others redirect to chat home. */
export function AdminRoute({ user, children }) {
  if (!user) return <Navigate to="/login" replace />
  if (!user.is_admin) return <Navigate to="/chat" replace />
  return children
}
