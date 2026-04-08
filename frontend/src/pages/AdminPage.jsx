import { useEffect, useState } from 'react'
import { apiFetch } from '../api/client'

export function AdminPage() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await apiFetch('/api/admin/insights')
        if (!cancelled) setData(res)
      } catch (e) {
        if (!cancelled) setError(e.data?.error || e.message)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="center-page wide">
      <h1>Admin insights</h1>
      {error ? <p className="error">{error}</p> : null}
      {data ? <pre className="json-preview">{JSON.stringify(data, null, 2)}</pre> : null}
    </div>
  )
}
