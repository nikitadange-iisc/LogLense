const BASE = '/api'

async function handleResponse(res) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const uploadLog = (file, dataset, onProgress) => {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)
    form.append('dataset', dataset)

    const xhr = new XMLHttpRequest()

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    })

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)) } catch { resolve({}) }
      } else {
        try {
          const err = JSON.parse(xhr.responseText)
          reject(new Error(err.detail || `HTTP ${xhr.status}`))
        } catch {
          reject(new Error(`HTTP ${xhr.status}`))
        }
      }
    })

    xhr.addEventListener('error', () => reject(new Error('Network error during upload')))
    xhr.addEventListener('abort', () => reject(new Error('Upload aborted')))

    xhr.open('POST', `${BASE}/upload`)
    xhr.send(form)
  })
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

export const deleteSession = (sessionId) =>
  fetch(`${BASE}/history/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  }).then(handleResponse)

export const renameSession = (sessionId, filename) =>
  fetch(`${BASE}/history/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename }),
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

export const getScores = () =>
  fetch(`${BASE}/scores`).then(handleResponse)

export const getLogs = (page = 0) =>
  fetch(`${BASE}/logs?page=${page}`).then(handleResponse)
