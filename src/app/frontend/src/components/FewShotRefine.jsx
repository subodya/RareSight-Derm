import { useState, useRef } from 'react'
import { ChevronDown, ChevronRight, Zap, CheckCircle2, X, Upload } from 'lucide-react'

export default function FewShotRefine({ sessionId, onSessionUpdate, classes }) {
  const [open, setOpen] = useState(false)
  const [selectedClassId, setSelectedClassId] = useState('')
  const [files, setFiles] = useState([])
  const [submitting, setSubmitting] = useState(false)
  const [refinedClasses, setRefinedClasses] = useState([])
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const classEntries = Object.entries(classes || {})

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files).slice(0, 5)
    setFiles(selected)
  }

  const removeFile = (i) => setFiles(f => f.filter((_, idx) => idx !== i))

  const handleSubmit = async () => {
    if (!selectedClassId || !files.length) return
    setSubmitting(true)
    setError(null)

    try {
      const fd = new FormData()
      fd.append('class_id', selectedClassId)
      if (sessionId) fd.append('session_id', sessionId)
      files.forEach(f => fd.append('files', f))

      const res = await fetch('/api/refine', { method: 'POST', body: fd })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Server error ${res.status}`)
      }
      const data = await res.json()
      onSessionUpdate(data.session_id)

      const cls = classes[selectedClassId]
      setRefinedClasses(prev => {
        const existing = prev.find(r => r.class_id === selectedClassId)
        if (existing) {
          return prev.map(r => r.class_id === selectedClassId ? { ...r, count: data.images_used } : r)
        }
        return [...prev, { class_id: selectedClassId, name: cls?.name || selectedClassId, count: data.images_used }]
      })

      setFiles([])
      setSelectedClassId('')
      if (inputRef.current) inputRef.current.value = ''
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  const removeRefined = (cls_id) => {
    setRefinedClasses(prev => prev.filter(r => r.class_id !== cls_id))
  }

  return (
    <div className="card" style={{ overflow: 'hidden', borderColor: open ? 'var(--accent)' : 'var(--border)', transition: 'border-color .2s' }}>
      {/* Toggle header */}
      <button
        style={{
          width: '100%', padding: '12px 16px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          color: open ? 'var(--accent)' : 'var(--muted)',
        }}
        onClick={() => setOpen(o => !o)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Zap size={13} />
          <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Advanced: Few-Shot Refinement
          </span>
          {refinedClasses.length > 0 && (
            <span style={{
              fontSize: 11, fontFamily: "'JetBrains Mono',monospace",
              padding: '2px 8px', borderRadius: 999,
              background: 'var(--accent-soft)', color: 'var(--accent)',
            }}>
              {refinedClasses.length} refined
            </span>
          )}
        </div>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>

      {open && (
        <div style={{ padding: '0 16px 16px', borderTop: '1px solid var(--border)' }}>
          <p style={{ fontSize: 12, marginTop: 12, marginBottom: 16, color: 'var(--muted)', lineHeight: 1.5 }}>
            Upload 1–5 reference images of a specific condition to dynamically teach the system. The next analysis will use your examples alongside the built-in prototype library.
          </p>

          {/* Active refined chips */}
          {refinedClasses.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
              {refinedClasses.map(r => (
                <div key={r.class_id} style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  fontSize: 12, padding: '4px 10px', borderRadius: 999,
                  background: 'var(--ok-bg)', border: '1px solid rgba(21,128,61,0.25)', color: 'var(--ok-ink)',
                }}>
                  <CheckCircle2 size={10} />
                  <span>{r.name}</span>
                  <span className="mono" style={{ color: 'var(--ok-ink)', opacity: 0.7 }}>×{r.count}</span>
                  <button onClick={() => removeRefined(r.class_id)} style={{ color: 'var(--ok-ink)' }}>
                    <X size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Class selector */}
          <div style={{ marginBottom: 12 }}>
            <label style={{ fontSize: 11, fontFamily: "'JetBrains Mono',monospace", display: 'block', marginBottom: 6, color: 'var(--muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              SELECT CONDITION
            </label>
            <select
              value={selectedClassId}
              onChange={e => setSelectedClassId(e.target.value)}
              style={{
                width: '100%', fontSize: 13, borderRadius: 10, padding: '10px 12px', outline: 'none',
                background: 'var(--surface)', border: '1px solid var(--border)', color: selectedClassId ? 'var(--ink)' : 'var(--muted)',
              }}
            >
              <option value="">Choose a condition...</option>
              {classEntries.map(([id, cls]) => (
                <option key={id} value={id}>{cls.name}</option>
              ))}
            </select>
          </div>

          {/* File upload */}
          <div style={{ marginBottom: 12 }}>
            <label style={{ fontSize: 11, fontFamily: "'JetBrains Mono',monospace", display: 'block', marginBottom: 6, color: 'var(--muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              REFERENCE IMAGES (max 5)
            </label>
            <div
              style={{
                borderRadius: 10, border: '1px dashed var(--border-strong)',
                padding: 12, cursor: 'pointer', background: 'var(--bg)',
              }}
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={handleFileChange}
              />
              {files.length === 0 ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Upload size={13} style={{ color: 'var(--muted)' }} />
                  <span style={{ fontSize: 12, color: 'var(--muted)' }}>Click to select images</span>
                </div>
              ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {files.map((f, i) => (
                    <div key={i} style={{
                      display: 'flex', alignItems: 'center', gap: 4, fontSize: 12,
                      padding: '3px 10px', borderRadius: 999,
                      background: 'var(--accent-soft)', color: 'var(--primary)',
                    }}>
                      <span style={{ maxWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                      <button onClick={e => { e.stopPropagation(); removeFile(i) }} style={{ color: 'var(--primary)' }}>
                        <X size={10} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {error && <p style={{ fontSize: 12, marginBottom: 10, color: 'var(--risk-ink)' }}>{error}</p>}

          <button
            onClick={handleSubmit}
            disabled={!selectedClassId || !files.length || submitting}
            className="btn btn-soft"
            style={{
              width: '100%', justifyContent: 'center', padding: '10px',
              opacity: selectedClassId && files.length && !submitting ? 1 : 0.5,
              cursor: !selectedClassId || !files.length ? 'not-allowed' : 'pointer',
            }}
          >
            {submitting ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ width: 12, height: 12, borderRadius: 999, border: '2px solid var(--primary)', borderTopColor: 'transparent' }} className="spin" />
                Teaching system...
              </span>
            ) : (
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Zap size={12} /> Teach System
              </span>
            )}
          </button>
        </div>
      )}
    </div>
  )
}
