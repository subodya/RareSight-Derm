import { useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { getCase, updateCase } from '../api'
import { X, Save, AlertTriangle } from '../icons'

const FIELDS = [
  { key: 'patient_name', label: 'Patient Name', type: 'text', placeholder: 'Unnamed Patient' },
  { key: 'age',          label: 'Age',          type: 'text', placeholder: 'e.g. 54' },
  { key: 'sex',          label: 'Sex',          type: 'select', options: [
      { value: '', label: '—' }, { value: 'M', label: 'Male' },
      { value: 'F', label: 'Female' }, { value: 'unknown', label: 'Unknown' },
    ] },
  { key: 'localization', label: 'Lesion Site',  type: 'text', placeholder: 'e.g. back, scalp' },
  { key: 'scan_type',    label: 'Scan Type',    type: 'text', placeholder: 'Dermoscopy' },
  { key: 'status',       label: 'Status',       type: 'select', options: [
      { value: 'pending', label: 'Pending' }, { value: 'confirmed', label: 'Confirmed' },
      { value: 'referred', label: 'Referred' },
    ] },
]

const FIELD_KEYS = FIELDS.map(f => f.key).concat('clinical_note')

export default function CaseEditModal({ caseId, onClose, onSaved }) {
  const [form, setForm]     = useState(null)
  const [error, setError]   = useState(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    getCase(caseId)
      .then(c => {
        const seed = {}
        FIELD_KEYS.forEach(k => { seed[k] = c[k] ?? '' })
        setForm(seed)
      })
      .catch(e => setError(e.message))
  }, [caseId])

  useEffect(() => {
    const h = e => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', h)
    return () => document.removeEventListener('keydown', h)
  }, [onClose])

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const save = async () => {
    setSaving(true); setError(null)
    try {
      await updateCase(caseId, form)
      onSaved?.()
      onClose()
    } catch (e) {
      setError(e.message)
      setSaving(false)
    }
  }

  return createPortal(
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(14,26,54,0.45)',
        backdropFilter: 'blur(4px)', zIndex: 110,
        display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
        padding: '40px 24px', overflowY: 'auto',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        className="fade-up"
        style={{
          background: 'var(--surface)', borderRadius: 20, maxWidth: 560, width: '100%',
          boxShadow: 'var(--shadow-lg)', position: 'relative', overflow: 'hidden',
        }}
      >
        <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: 'var(--ink)' }}>Edit Case</h2>
            <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 2 }}>
              Patient & administrative details. AI prediction is not editable.
            </p>
          </div>
          <button
            onClick={onClose}
            style={{ width: 34, height: 34, borderRadius: 10, background: 'var(--bg-2)', color: 'var(--ink)', display: 'grid', placeItems: 'center' }}
          >
            <X size={15} />
          </button>
        </div>

        {!form && !error && (
          <div style={{ padding: 24 }}>
            <div className="shimmer" style={{ height: 180, borderRadius: 12 }} />
          </div>
        )}

        {form && (
          <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
              {FIELDS.map(f => (
                <label key={f.key} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--muted)' }}>
                    {f.label}
                  </span>
                  {f.type === 'select' ? (
                    <select className="input" value={form[f.key] || ''} onChange={e => set(f.key, e.target.value)}>
                      {f.options.map(o => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  ) : (
                    <input className="input" value={form[f.key] || ''} placeholder={f.placeholder}
                      onChange={e => set(f.key, e.target.value)} />
                  )}
                </label>
              ))}
            </div>

            <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--muted)' }}>
                Clinical Note
              </span>
              <textarea className="input" rows={4} value={form.clinical_note || ''}
                placeholder="Observations, history, follow-up plan…"
                onChange={e => set('clinical_note', e.target.value)}
                style={{ resize: 'vertical', fontFamily: 'inherit' }} />
            </label>

            {error && (
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 12, color: 'var(--risk-ink)' }}>
                <AlertTriangle size={14} /> {error}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 4 }}>
              <button onClick={onClose} className="btn"
                style={{ background: 'var(--bg-2)', color: 'var(--ink-2)' }}>
                Cancel
              </button>
              <button onClick={save} disabled={saving} className="btn btn-primary"
                style={{ opacity: saving ? 0.7 : 1 }}>
                <Save size={14} /> {saving ? 'Saving…' : 'Save Changes'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}
