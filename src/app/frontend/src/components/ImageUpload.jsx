import { useState, useRef, useCallback } from 'react'
import { Upload, Image as ImageIcon, CheckCircle2, X } from 'lucide-react'

export default function ImageUpload({ onImageSelect, selectedFile, previewUrl }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    onImageSelect(file, url)
  }, [onImageSelect])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onChange = (e) => handleFile(e.target.files[0])

  const clearFile = (e) => {
    e.stopPropagation()
    onImageSelect(null, null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  return (
    <div
      className="card"
      style={{
        overflow: 'hidden',
        borderColor: dragging ? 'var(--accent)' : 'var(--border)',
        boxShadow: dragging ? '0 0 0 3px var(--accent-soft)' : 'var(--shadow-sm)',
        transition: 'border-color .2s, box-shadow .2s',
      }}
    >
      <div
        className="relative cursor-pointer select-none"
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={onChange}
        />

        {selectedFile ? (
          <div style={{ padding: 16 }}>
            <div style={{ position: 'relative', display: 'flex', justifyContent: 'center', marginBottom: 12 }}>
              <div style={{ position: 'relative', borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)' }}>
                <img
                  src={previewUrl || ''}
                  alt="Patient scan preview"
                  style={{ width: 200, height: 200, objectFit: 'cover', display: 'block' }}
                  onError={(e) => { e.target.style.display = 'none' }}
                />
              </div>
              <button
                onClick={clearFile}
                style={{
                  position: 'absolute', top: 4, right: 4,
                  width: 24, height: 24, borderRadius: 999,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: 'var(--risk-ink)', color: '#fff',
                }}
              >
                <X size={12} />
              </button>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, marginBottom: 4 }}>
                <CheckCircle2 size={13} style={{ color: 'var(--ok-ink)' }} />
                <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--ok-ink)' }}>Ready for analysis</span>
              </div>
              <p className="mono" style={{ fontSize: 11, color: 'var(--muted)', maxWidth: 240, margin: '0 auto', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {selectedFile.name}
              </p>
              <p className="mono" style={{ fontSize: 11, color: 'var(--muted-2)', marginTop: 2 }}>
                {formatSize(selectedFile.size)}
              </p>
            </div>
          </div>
        ) : (
          <div style={{ padding: 32, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
            <div style={{
              width: 56, height: 56, borderRadius: 14,
              background: dragging ? 'var(--accent-soft)' : 'var(--bg-2)',
              border: '1px dashed',
              borderColor: dragging ? 'var(--accent)' : 'var(--border-strong)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all .2s',
            }}>
              {dragging
                ? <ImageIcon size={24} style={{ color: 'var(--accent)' }} />
                : <Upload size={24} style={{ color: 'var(--primary)' }} />
              }
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: 14, fontWeight: 700, color: dragging ? 'var(--accent)' : 'var(--ink)', marginBottom: 4 }}>
                {dragging ? 'Release to upload' : 'Drop dermoscopy image here'}
              </p>
              <p style={{ fontSize: 12, color: 'var(--muted)' }}>or click to browse · JPG, PNG, JPEG</p>
            </div>
            <div style={{ width: '100%', borderTop: '1px dashed var(--border)' }} />
            <p className="mono" style={{ fontSize: 10, color: 'var(--muted-2)', letterSpacing: '0.08em' }}>
              RECOMMENDED: dermoscopy · min 224×224
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
