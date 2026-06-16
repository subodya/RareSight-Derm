// RareSight API client — one function per backend domain endpoint.
// All paths are relative; Vite proxies /api → http://localhost:8000.

async function getJSON(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Request failed (${res.status})`)
  return res.json()
}

// ── System ──
export const getHealth  = () => getJSON('/api/health')
export const getClasses = () => getJSON('/api/classes')

// ── Dashboard ──
export const getDashboard = () => getJSON('/api/dashboard/summary')

// ── Cases ──
export const getCases       = (limit = 50) => getJSON(`/api/cases?limit=${limit}`)
export const getCase        = id => getJSON(`/api/cases/${id}`)
export const caseReportUrl  = id => `/api/cases/${id}/report`

export async function updateCase(id, fields) {
  const res = await fetch(`/api/cases/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(fields),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Update failed (${res.status})`)
  }
  return res.json()
}

export async function deleteCase(id) {
  const res = await fetch(`/api/cases/${id}`, { method: 'DELETE' })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Delete failed (${res.status})`)
  }
  return res.json()
}

// ── Library ──
export const getDiseases = () => getJSON('/api/library/diseases')

// ── Profile ──
export const getProfile = () => getJSON('/api/profile')

// ── Scan ──
export async function predict(formData) {
  const res = await fetch('/api/predict', { method: 'POST', body: formData })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Server error ${res.status}`)
  }
  return res.json()
}

export async function refine(formData) {
  const res = await fetch('/api/refine', { method: 'POST', body: formData })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Server error ${res.status}`)
  }
  return res.json()
}

// ── Helpers ──
export function timeAgo(iso) {
  if (!iso) return ''
  const then = new Date(iso)
  const mins = Math.floor((Date.now() - then.getTime()) / 60000)
  if (mins < 1) return 'Just now'
  if (mins < 60) return `${mins} min ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs} hour${hrs > 1 ? 's' : ''} ago`
  const days = Math.floor(hrs / 24)
  if (days === 1) return 'Yesterday'
  if (days < 7) return `${days} days ago`
  return then.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })
}
