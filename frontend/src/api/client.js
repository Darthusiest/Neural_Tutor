/**
 * API base: empty string uses Vite dev proxy; in prod set VITE_API_BASE_URL.
 */
const base = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? ''

export async function apiFetch(path, options = {}) {
  const url = `${base}${path.startsWith('/') ? path : `/${path}`}`
  const headers = { ...(options.headers || {}) }
  if (
    options.body &&
    typeof options.body === 'string' &&
    !headers['Content-Type']
  ) {
    headers['Content-Type'] = 'application/json'
  }
  const res = await fetch(url, {
    ...options,
    headers,
    credentials: 'include',
  })
  const text = await res.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = { raw: text }
  }
  if (!res.ok) {
    const err = new Error(data?.error || res.statusText || 'Request failed')
    err.status = res.status
    err.data = data
    throw err
  }
  return data
}
