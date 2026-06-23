const BASE = '/api'

async function handleResponse(res) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const uploadLog = (file, dataset) => {
  const form = new FormData()
  form.append('file', file)
  form.append('dataset', dataset)
  return fetch(`${BASE}/upload`, { method: 'POST', body: form }).then(handleResponse)
}

export const getStatus = () =>
  fetch(`${BASE}/status`).then(handleResponse)

export const cancelPipeline = () =>
  fetch(`${BASE}/pipeline`, { method: 'DELETE' }).then(handleResponse)

export const resetPipeline = () =>
  fetch(`${BASE}/reset`, { method: 'POST' }).then(handleResponse)

export const tryout = () =>
  fetch(`${BASE}/tryout`, { method: 'POST' }).then(handleResponse)

export const getHistory = () =>
  fetch(`${BASE}/history`).then(handleResponse)

export const activateSession = (sessionId) =>
  fetch(`${BASE}/history/${encodeURIComponent(sessionId)}/activate`, {
    method: 'POST',
  }).then(handleResponse)

export const getSessions = () =>
  fetch(`${BASE}/sessions`).then(handleResponse)

export const getSession = (id) =>
  fetch(`${BASE}/sessions/${encodeURIComponent(id)}`).then(handleResponse)

export const analyzeSession = (sessionId, topK = 3) =>
  fetch(`${BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, top_k: topK }),
  }).then(handleResponse)

export const chat = (question, sessionId = null, history = []) =>
  fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId, history }),
  }).then(handleResponse)
