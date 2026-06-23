import { useCallback, useState } from 'react'

const ACCEPTED = ['.log', '.txt', '.csv']

export default function FileUpload({ onFileSelect, disabled }) {
  const [dragging, setDragging] = useState(false)
  const [file, setFile] = useState(null)

  const accept = (f) => {
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
    if (!ACCEPTED.includes(ext)) {
      alert(`Unsupported file type "${ext}". Please upload a .log, .txt, or .csv file.`)
      return
    }
    setFile(f)
    onFileSelect(f)
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    if (disabled) return
    const f = e.dataTransfer.files[0]
    if (f) accept(f)
  }, [disabled])

  const onDragOver = (e) => { e.preventDefault(); if (!disabled) setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const onChange = (e) => {
    const f = e.target.files[0]
    if (f) accept(f)
    e.target.value = ''
  }

  return (
    <label
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={`
        relative flex flex-col items-center justify-center gap-4
        w-full h-48 rounded-xl border-2 border-dashed cursor-pointer
        transition-colors select-none
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${dragging
          ? 'border-blue-400 bg-blue-50 dark:bg-blue-950/40'
          : file
            ? 'border-green-500 bg-green-50 dark:bg-green-950/20'
            : 'border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800/50 hover:border-gray-400 dark:hover:border-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800'
        }
      `}
    >
      <input
        type="file"
        accept=".log,.txt,.csv"
        className="sr-only"
        onChange={onChange}
        disabled={disabled}
      />

      {file ? (
        <>
          <svg className="w-10 h-10 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="text-center">
            <p className="text-green-400 font-medium">{file.name}</p>
            <p className="text-gray-500 text-sm mt-1">{(file.size / 1024).toFixed(1)} KB</p>
          </div>
        </>
      ) : (
        <>
          <svg className="w-10 h-10 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <div className="text-center">
            <p className="text-gray-700 dark:text-gray-300 font-medium">Drop your log file here</p>
            <p className="text-gray-500 text-sm mt-1">or click to browse &nbsp;·&nbsp; .log .txt .csv</p>
          </div>
        </>
      )}
    </label>
  )
}
