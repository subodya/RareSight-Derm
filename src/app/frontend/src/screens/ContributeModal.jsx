import { useState, useEffect } from 'react'
import { PlaceholderImage } from '../components/common'
import { Upload, Book, Image, Shield, CheckCircle, Check, X, Info, ChevronRight, Plus, Clock, ArrowRight } from '../icons'

export default function ContributeModal({ onClose }) {
  const [step, setStep] = useState(0)
  const [form, setForm] = useState({
    name: '',
    code: '',
    category: 'Malignant',
    modality: 'Dermoscopy',
    riskFactors: '',
    age: '',
    prevalence: '',
    summary: '',
    diagnosticCriteria: '',
    references: '',
    anonConfirmed: false,
    consentConfirmed: false,
    images: [],
  })

  useEffect(() => {
    const h = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', h)
    return () => document.removeEventListener('keydown', h)
  }, [onClose])

  const update = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const canAdvance = (() => {
    if (step === 1) return form.name.trim() && form.code.trim() && form.summary.trim()
    if (step === 2) return form.images.length >= 3
    if (step === 3) return form.diagnosticCriteria.trim() && form.anonConfirmed && form.consentConfirmed
    return true
  })()

  const STEPS = [
    { label: 'Overview',           Icon: Info },
    { label: 'Disease Info',       Icon: Book },
    { label: 'Reference Images',   Icon: Image },
    { label: 'Evidence & Consent', Icon: Shield },
    { label: 'Review',             Icon: CheckCircle },
  ]

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(14,26,54,0.50)', backdropFilter: 'blur(6px)',
        zIndex: 100, display: 'grid', placeItems: 'center', padding: 24,
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        className="fade-up"
        style={{
          background: 'var(--surface)', borderRadius: 22,
          maxWidth: 960, width: '100%', maxHeight: '92vh',
          display: 'grid', gridTemplateColumns: '240px 1fr',
          boxShadow: 'var(--shadow-lg)', overflow: 'hidden',
        }}
      >
        {/* Left rail */}
        <div style={{
          background: 'linear-gradient(170deg, var(--primary) 0%, #2A4FAE 100%)',
          color: '#fff', padding: '28px 22px',
          display: 'flex', flexDirection: 'column', gap: 24,
        }}>
          <div>
            <div style={{ width: 38, height: 38, borderRadius: 10, background: 'rgba(255,255,255,0.15)', display: 'grid', placeItems: 'center', marginBottom: 14 }}>
              <Upload size={18} strokeWidth={1.7} />
            </div>
            <h2 style={{ fontSize: 18, fontWeight: 700, letterSpacing: '-0.01em' }}>Contribute Data</h2>
            <p style={{ fontSize: 12, color: 'rgba(255,255,255,0.72)', marginTop: 4, lineHeight: 1.5 }}>
              Submit a candidate disease for clinical verification.
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {STEPS.map((s, i) => {
              const done = i < step
              const current = i === step
              const IconC = s.Icon
              return (
                <div key={s.label} style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 12px', borderRadius: 10,
                  background: current ? 'rgba(255,255,255,0.16)' : 'transparent',
                  opacity: done || current ? 1 : 0.55,
                }}>
                  <div style={{
                    width: 22, height: 22, borderRadius: 999, display: 'grid', placeItems: 'center',
                    background: done ? '#fff' : 'rgba(255,255,255,0.14)',
                    color: done ? 'var(--primary)' : '#fff',
                    fontSize: 11, fontWeight: 700, flexShrink: 0,
                  }}>
                    {done ? <Check size={12} strokeWidth={2.4} /> : i + 1}
                  </div>
                  <span style={{ fontSize: 12, fontWeight: current ? 600 : 500 }}>{s.label}</span>
                </div>
              )
            })}
          </div>

          <div style={{ marginTop: 'auto', fontSize: 11, color: 'rgba(255,255,255,0.55)', lineHeight: 1.55 }}>
            Submissions reviewed by the RareSight clinical advisory board within 5–7 business days.
          </div>
        </div>

        {/* Right content */}
        <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '18px 24px', borderBottom: '1px solid var(--border)' }}>
            <div>
              <div className="mono" style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                Step {Math.min(step + 1, STEPS.length)} of {STEPS.length}
              </div>
              <h3 style={{ fontSize: 17, fontWeight: 700, marginTop: 2 }}>{STEPS[step].label}</h3>
            </div>
            <button onClick={onClose} style={{ width: 32, height: 32, borderRadius: 8, display: 'grid', placeItems: 'center', color: 'var(--muted)', background: 'var(--bg-2)' }}>
              <X size={16} />
            </button>
          </div>

          <div style={{ flex: 1, overflow: 'auto', padding: '24px 28px' }} className="fade-up" key={step}>
            {step === 0 && <Overview />}
            {step === 1 && <DiseaseInfoStep form={form} update={update} />}
            {step === 2 && <ImagesStep form={form} update={update} />}
            {step === 3 && <EvidenceStep form={form} update={update} />}
            {step === 4 && <ReviewStep form={form} />}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 24px', borderTop: '1px solid var(--border)', background: 'var(--bg)' }}>
            <button className="btn btn-ghost" onClick={() => step === 0 ? onClose() : setStep(step - 1)}>
              {step === 0 ? 'Cancel' : <><ChevronRight size={14} style={{ transform: 'rotate(180deg)' }} /> Back</>}
            </button>
            {step < STEPS.length - 1 ? (
              <button
                className="btn btn-primary"
                disabled={!canAdvance}
                onClick={() => setStep(step + 1)}
                style={{ opacity: canAdvance ? 1 : 0.5, cursor: canAdvance ? 'pointer' : 'not-allowed' }}
              >
                {step === 0 ? 'Start Submission' : 'Continue'} <ChevronRight size={14} />
              </button>
            ) : (
              <SubmittedView onClose={onClose} />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function Overview() {
  return (
    <div>
      <p style={{ fontSize: 14, color: 'var(--ink-2)', lineHeight: 1.6, marginBottom: 20 }}>
        Help expand the global RareSight knowledge base. Submitted diseases enter the verification pipeline, reviewed by the clinical advisory board.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
        <InfoTile Icon={Book}        title="Disease Info"      sub="Name, ICD-10 code, clinical summary, risk factors, prevalence." />
        <InfoTile Icon={Image}       title="Reference Images"  sub="Upload 3–10 anonymized dermoscopy scans showing typical presentation." />
        <InfoTile Icon={Shield}      title="Evidence"          sub="Diagnostic criteria, citations, patient consent confirmation." />
        <InfoTile Icon={CheckCircle} title="Verification"      sub="Reviewed within 5–7 days. You'll be credited as a contributor." />
      </div>
      <div style={{ padding: 14, borderRadius: 12, background: 'var(--accent-soft)', display: 'flex', gap: 10, alignItems: 'flex-start' }}>
        <Info size={16} style={{ color: 'var(--primary)', flexShrink: 0, marginTop: 1 }} />
        <div style={{ fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.5 }}>
          <strong style={{ color: 'var(--primary)' }}>Verified submissions</strong> are added to the public Library and used to retrain the prototype set after manual review.
        </div>
      </div>
    </div>
  )
}

function InfoTile({ Icon: IconC, title, sub }) {
  return (
    <div style={{ padding: 14, borderRadius: 12, background: 'var(--bg)', border: '1px solid var(--border)' }}>
      <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--accent-soft)', color: 'var(--primary)', display: 'grid', placeItems: 'center', marginBottom: 10 }}>
        <IconC size={15} />
      </div>
      <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--ink)', marginBottom: 3 }}>{title}</div>
      <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.5 }}>{sub}</div>
    </div>
  )
}

const inputStyle = {
  width: '100%', padding: '10px 14px', borderRadius: 10,
  border: '1px solid var(--border)', background: 'var(--surface)',
  fontSize: 13, color: 'var(--ink)', outline: 'none', fontFamily: 'inherit',
}

function FormField({ label, required, children }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
        {label}{required && <span style={{ color: 'var(--risk-ink)', marginLeft: 4 }}>*</span>}
      </span>
      {children}
    </label>
  )
}

function DiseaseInfoStep({ form, update }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12 }}>
        <FormField label="Disease name" required>
          <input className="cm-input" placeholder="e.g. Melanoma" value={form.name} onChange={e => update('name', e.target.value)} />
        </FormField>
        <FormField label="ICD-10 code" required>
          <input className="cm-input mono" placeholder="C43.9" value={form.code} onChange={e => update('code', e.target.value)} />
        </FormField>
      </div>

      <FormField label="Clinical summary" required>
        <textarea className="cm-input" rows="3" placeholder="Brief description of the disease, typical presentation, and key distinguishing features…"
          value={form.summary} onChange={e => update('summary', e.target.value)} style={{ resize: 'vertical', minHeight: 84 }} />
      </FormField>

      <FormField label="Category (maps to Library filters)" required>
        <div style={{ display: 'flex', gap: 8 }}>
          {['Malignant', 'Benign', 'Precancerous'].map(c => (
            <button key={c} type="button" onClick={() => update('category', c)} style={{
              flex: 1, padding: '10px 14px', borderRadius: 10, textAlign: 'left', fontSize: 13, fontWeight: 600,
              border: `1px solid ${form.category === c ? 'var(--primary)' : 'var(--border)'}`,
              background: form.category === c ? 'var(--accent-soft)' : 'var(--surface)',
              color: form.category === c ? 'var(--primary)' : 'var(--ink)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                {form.category === c && <Check size={13} strokeWidth={2.4} />} {c}
              </div>
              <div style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 500 }}>
                {c === 'Malignant' && 'Cancerous neoplasm'}
                {c === 'Benign' && 'Non-cancerous lesion'}
                {c === 'Precancerous' && 'Potential malignant precursor'}
              </div>
            </button>
          ))}
        </div>
      </FormField>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        <FormField label="Modality">
          <select className="cm-input" value={form.modality} onChange={e => update('modality', e.target.value)}>
            <option>Dermoscopy</option>
            <option>Clinical Photo</option>
            <option>Histopathology</option>
            <option>Other</option>
          </select>
        </FormField>
        <FormField label="Typical onset">
          <input className="cm-input" placeholder="e.g. 40–70 yrs" value={form.age} onChange={e => update('age', e.target.value)} />
        </FormField>
        <FormField label="Estimated prevalence">
          <input className="cm-input" placeholder="e.g. 1:10,000" value={form.prevalence} onChange={e => update('prevalence', e.target.value)} />
        </FormField>
      </div>

      <FormField label="Primary risk factors">
        <input className="cm-input" placeholder="e.g. UV exposure, fair skin, immunosuppression" value={form.riskFactors} onChange={e => update('riskFactors', e.target.value)} />
      </FormField>
    </div>
  )
}

function ImagesStep({ form, update }) {
  const handleAdd = () => {
    update('images', [...form.images, { id: Date.now() + Math.random(), name: `scan_${form.images.length + 1}.png`, kb: 1800 + Math.round(Math.random() * 1200), seed: form.images.length }])
  }
  const handleRemove = (id) => update('images', form.images.filter(i => i.id !== id))

  return (
    <div>
      <p style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 14, lineHeight: 1.5 }}>
        Upload 3–10 anonymized reference images showing typical presentation. <strong style={{ color: 'var(--ink)' }}>All identifying metadata is automatically stripped.</strong>
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 14 }}>
        {form.images.map((img, i) => (
          <div key={img.id} style={{ position: 'relative' }}>
            <PlaceholderImage label="" height={110} seed={img.seed + 17} radius={10} />
            {i === 0 && <span className="pill" style={{ position: 'absolute', top: 6, left: 6, background: 'var(--primary)', color: '#fff', fontSize: 9, padding: '2px 7px' }}>Primary</span>}
            <button onClick={() => handleRemove(img.id)} style={{
              position: 'absolute', top: 6, right: 6, width: 22, height: 22, borderRadius: 999,
              background: 'rgba(255,255,255,0.95)', color: 'var(--risk-ink)', display: 'grid', placeItems: 'center',
            }}>
              <X size={12} strokeWidth={2.2} />
            </button>
          </div>
        ))}
        {form.images.length < 10 && (
          <button onClick={handleAdd} style={{
            aspectRatio: '1', border: '2px dashed var(--border-strong)', background: 'var(--bg)', borderRadius: 10,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 4, color: 'var(--primary)',
          }}>
            <Plus size={20} />
            <span style={{ fontSize: 10, fontWeight: 600 }}>Add image</span>
          </button>
        )}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--muted)' }}>
        <span>
          <strong style={{ color: form.images.length >= 3 ? 'var(--ok-ink)' : 'var(--risk-ink)' }}>{form.images.length}</strong> / 10 uploaded
          {form.images.length < 3 && <span style={{ marginLeft: 8 }}>· Min 3 required</span>}
        </span>
        <span className="mono">Dermoscopy · PNG · JPG · up to 25 MB each</span>
      </div>
    </div>
  )
}

function EvidenceStep({ form, update }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <FormField label="Diagnostic criteria" required>
        <textarea className="cm-input" rows="4" placeholder="List the diagnostic criteria (clinical signs, imaging findings, lab markers, genetic confirmation)…"
          value={form.diagnosticCriteria} onChange={e => update('diagnosticCriteria', e.target.value)} style={{ resize: 'vertical', minHeight: 100 }} />
      </FormField>
      <FormField label="Supporting references (optional)">
        <textarea className="cm-input" rows="3" placeholder="DOIs, PubMed IDs, OMIM entries — one per line"
          value={form.references} onChange={e => update('references', e.target.value)}
          style={{ resize: 'vertical', minHeight: 76, fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }} />
      </FormField>
      <div style={{ padding: 16, borderRadius: 12, background: 'var(--bg)', border: '1px solid var(--border)' }}>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Verification & Compliance</div>
        <ConsentCheck checked={form.anonConfirmed} onChange={() => update('anonConfirmed', !form.anonConfirmed)}
          label="Patient-identifying information removed"
          sub="All EXIF data, names, dates of birth, and clinic identifiers have been stripped from the uploaded files." />
        <ConsentCheck checked={form.consentConfirmed} onChange={() => update('consentConfirmed', !form.consentConfirmed)}
          label="Patient consent obtained"
          sub="Documented written consent for anonymized clinical use, in accordance with local IRB or equivalent." />
      </div>
    </div>
  )
}

function ConsentCheck({ checked, onChange, label, sub }) {
  return (
    <button onClick={onChange} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, padding: '12px 0', width: '100%', textAlign: 'left', borderTop: '1px dashed var(--border)' }}>
      <div style={{
        width: 20, height: 20, borderRadius: 6, flexShrink: 0, marginTop: 1,
        background: checked ? 'var(--primary)' : 'var(--surface)',
        border: `1.5px solid ${checked ? 'var(--primary)' : 'var(--border-strong)'}`,
        display: 'grid', placeItems: 'center', color: '#fff',
      }}>
        {checked && <Check size={13} strokeWidth={2.6} />}
      </div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>{label}</div>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2, lineHeight: 1.5 }}>{sub}</div>
      </div>
    </button>
  )
}

function ReviewStep({ form }) {
  return (
    <div>
      <div style={{ padding: 18, borderRadius: 14, background: 'linear-gradient(155deg, var(--primary) 0%, #2A4FAE 100%)', color: '#fff', marginBottom: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
          <div>
            <span className="pill" style={{ background: 'rgba(255,255,255,0.18)', color: '#fff', marginBottom: 8 }}>{form.category}</span>
            <h3 style={{ fontSize: 20, fontWeight: 700, marginTop: 6, letterSpacing: '-0.01em' }}>{form.name || 'Untitled Disease'}</h3>
            <div className="mono" style={{ fontSize: 11, color: 'rgba(255,255,255,0.7)', marginTop: 3 }}>ICD-10 · {form.code || '—'} · {form.modality}</div>
          </div>
          <span className="pill" style={{ background: 'rgba(255,255,255,0.18)', color: '#fff', display: 'inline-flex', gap: 6 }}>
            <Clock size={11} /> Pending Review
          </span>
        </div>
        <p style={{ fontSize: 13, color: 'rgba(255,255,255,0.85)', lineHeight: 1.55 }}>{form.summary || '—'}</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 18 }}>
        {[['Risk Factors', form.riskFactors || '—'], ['Onset', form.age || '—'], ['Prevalence', form.prevalence || '—']].map(([l, v]) => (
          <div key={l} style={{ padding: '10px 12px', borderRadius: 10, background: 'var(--bg)' }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>{l}</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)', marginTop: 4 }}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', padding: 14, borderRadius: 12, background: 'var(--ok-bg)' }}>
        <CheckCircle size={16} style={{ color: 'var(--ok-ink)', flexShrink: 0, marginTop: 1 }} />
        <div style={{ fontSize: 12, color: 'var(--ok-ink)', lineHeight: 1.55 }}>
          Ready to submit. After review, this disease will appear in the Library under the <strong>{form.category}</strong> filter with full attribution.
        </div>
      </div>
    </div>
  )
}

function SubmittedView({ onClose }) {
  const [submitted, setSubmitted] = useState(false)
  const submit = () => { setSubmitted(true); setTimeout(onClose, 2400) }
  if (submitted) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--ok-ink)', fontSize: 13, fontWeight: 600 }}>
        <CheckCircle size={18} /> Submitted! Reference ID <span className="mono">RS-{Math.floor(Math.random() * 90000 + 10000)}</span>
      </div>
    )
  }
  return (
    <button className="btn btn-primary" onClick={submit}>
      <Upload size={14} /> Submit for Verification
    </button>
  )
}
