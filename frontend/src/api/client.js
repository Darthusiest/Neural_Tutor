/**
 * API base: empty string uses Vite proxy to Flask (same origin /api).
 */
const base = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? ''

let csrfToken = null

async function fetchCsrfToken() {
  const url = `${base}/api/auth/csrf`
  const res = await fetch(url, { credentials: 'include' })
  const text = await res.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = { raw: text }
  }
  if (!res.ok) {
    const err = new Error(data?.error || res.statusText || 'CSRF fetch failed')
    err.status = res.status
    err.data = data
    throw err
  }
  if (!data?.csrf_token) {
    throw new Error('Missing csrf_token from server')
  }
  csrfToken = data.csrf_token
  return csrfToken
}

export function invalidateCsrf() {
  csrfToken = null
}

/**
 * @param {string} path
 * @param {RequestInit} [options]
 * @param {boolean} [isRetry]
 */
export async function apiFetch(path, options = {}, isRetry = false) {
  const url = `${base}${path.startsWith('/') ? path : `/${path}`}`
  const method = (options.method || 'GET').toUpperCase()
  const headers = { ...(options.headers || {}) }

  if (
    ['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)
  ) {
    if (!csrfToken) {
      await fetchCsrfToken()
    }
    headers['X-CSRFToken'] = csrfToken
  }

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

  if (
    res.status === 403 &&
    data?.error === 'csrf validation failed' &&
    !isRetry &&
    ['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)
  ) {
    invalidateCsrf()
    return apiFetch(path, options, true)
  }

  if (!res.ok) {
    const err = new Error(data?.error || res.statusText || 'Request failed')
    err.status = res.status
    err.data = data
    throw err
  }
  return data
}
