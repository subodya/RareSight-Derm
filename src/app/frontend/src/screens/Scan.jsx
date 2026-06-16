import { useState, useCallback } from 'react'
import { Upload, Cpu, Check, ArrowUpRight, Microscope, Activity, Sparkles,
  AlertTriangle, Stethoscope, Refresh, Download,
  CheckCircle, Info } from '../icons'
import { SectionHeader, RiskPill } from '../components/common'
import ImageUpload from '../components/ImageUpload'
import DiagnosisResults from '../components/DiagnosisResults'
import AttentionHeatmap from '../components/AttentionHeatmap'
import ReferenceGallery from '../components/ReferenceGallery'
import FewShotRefine from '../components/FewShotRefine'
import { ANALYSIS_STEPS } from '../data'
import { predict, caseReportUrl } from '../api'

export default function Scan({ classes, systemStatus }) {
  const [uploadedFile, setUploadedFile]   = useState(null)
  const [previewUrl, setPreviewUrl]       = useState(null)
  const [loading, setLoading]             = useState(false)
  const [analysisStep, setAnalysisStep]   = useState(null)
  const [result, setResult]               = useState(null)
  const [error, setError]                 = useState(null)
  const [sessionId, setSessionId]         = useState(null)
  const [patientName, setPatientName]     = useState('')
  const [patientAge, setPatientAge]       = useState('')
  const [patientSex, setPatientSex]       = useState('')
  const [localization, setLocalization]   = useState('')
  const [scanType, setScanType]           = useState('Dermoscopy')
  const [clinicalNotes, setClinicalNotes] = useState('')

  const handleImageSelect = useCallback((file, url) => {
    setUploadedFile(file)
    setPreviewUrl(url)
    setResult(null)
    setError(null)
    setAnalysisStep(null)
  }, [])

  const handleReset = () => {
    setUploadedFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    setPatientName('')
    setClinicalNotes('')
  }

  const handleAnalyze = async () => {
    if (!uploadedFile) return
    setLoading(true)
    setError(null)
    setResult(null)

    let stepIdx = 0
    setAnalysisStep(ANALYSIS_STEPS[0])
    const interval = setInterval(() => {
      stepIdx = Math.min(stepIdx + 1, ANALYSIS_STEPS.length - 1)
      setAnalysisStep(ANALYSIS_STEPS[stepIdx])
    }, 600)

    try {
      const fd = new FormData()
      fd.append('file', uploadedFile)
      if (sessionId) fd.append('session_id', sessionId)
      if (patientName)   fd.append('patient_name', patientName)
      if (patientAge)    fd.append('age', patientAge)
      if (patientSex)    fd.append('sex', patientSex)
      if (localization)  fd.append('localization', localization)
      if (scanType)      fd.append('scan_type', scanType)
      if (clinicalNotes) fd.append('clinical_note', clinicalNotes)

      const data = await predict(fd)
      setResult(data)
    } catch (e) {
      setError(e.message || 'Analysis failed. Check that the backend is running.')
    } finally {
      clearInterval(interval)
      setLoading(false)
      setAnalysisStep(null)
    }
  }

  const stepActive = result ? 3 : (loading ? 2 : (uploadedFile ? 2 : 1))

  return (
    <div className="page fade-up">
      <StepFlow active={stepActive} />

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 3fr', gap: 24, marginTop: 28 }}>
        {/* LEFT: Intake */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <SectionHeader icon={<Activity size={14} />} label="Patient Intake" step="01" />

          <ImageUpload
            onImageSelect={handleImageSelect}
            selectedFile={uploadedFile}
            previewUrl={previewUrl}
          />

          <div className="card" style={{ padding: 18 }}>
            <h3 style={{ fontSize: 13, fontWeight: 700, marginBottom: 14, color: 'var(--ink)' }}>Patient Details</h3>
            <div style={{ marginBottom: 12 }}>
              <FormField label="Patient Name">
                <input
                  type="text" placeholder="Optional — shown on dashboard & report" value={patientName}
                  onChange={e => setPatientName(e.target.value)} style={inputStyle}
                />
              </FormField>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <FormField label="Age">
                <input
                  type="text" placeholder="45" value={patientAge}
                  onChange={e => setPatientAge(e.target.value)} style={inputStyle}
                />
              </FormField>
              <FormField label="Sex">
                <select value={patientSex} onChange={e => setPatientSex(e.target.value)} style={inputStyle}>
                  <option value="">Select</option>
                  <option value="F">Female</option>
                  <option value="M">Male</option>
                  <option value="O">Other</option>
                </select>
              </FormField>
            </div>
            <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <FormField label="Anatomical Site">
                <select value={localization} onChange={e => setLocalization(e.target.value)} style={inputStyle}>
                  <option value="">Select</option>
                  {ANATOMICAL_SITES.map(s => (
                    <option key={s} value={s}>{s.replace(/\b\w/g, c => c.toUpperCase())}</option>
                  ))}
                </select>
              </FormField>
              <FormField label="Scan Modality">
                <select value={scanType} onChange={e => setScanType(e.target.value)} style={inputStyle}>
                  <option>Dermoscopy</option>
                  <option>Clinical Photo</option>
                  <option>Histopathology</option>
                </select>
              </FormField>
            </div>
            <div style={{ marginTop: 12 }}>
              <FormField label="Clinical Notes">
                <textarea
                  rows="3"
                  placeholder="Symptoms, history, presenting complaint…"
                  value={clinicalNotes}
                  onChange={e => setClinicalNotes(e.target.value)}
                  style={{ ...inputStyle, resize: 'vertical', minHeight: 80, fontFamily: 'inherit' }}
                />
              </FormField>
            </div>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!uploadedFile || loading || systemStatus !== 'online'}
            className="btn btn-primary"
            style={{
              padding: '14px 18px', fontSize: 14, justifyContent: 'center',
              opacity: uploadedFile && !loading && systemStatus === 'online' ? 1 : 0.5,
              cursor: !uploadedFile || loading || systemStatus !== 'online' ? 'not-allowed' : 'pointer',
            }}
          >
            {loading ? (
              <>
                <Refresh size={15} className="spin" />
                <span className="mono" style={{ fontSize: 11 }}>{analysisStep || 'Analyzing…'}</span>
              </>
            ) : (
              <><Sparkles size={15} strokeWidth={1.8} /> Run Clinical Analysis</>
            )}
          </button>

          {result && (
            <FewShotRefine
              sessionId={sessionId}
              onSessionUpdate={setSessionId}
              classes={classes}
            />
          )}

          <div style={{
            fontSize: 11, color: 'var(--muted)', display: 'flex', gap: 8,
            padding: '10px 14px', borderRadius: 10, background: 'var(--bg-2)', lineHeight: 1.5,
          }}>
            <Stethoscope size={14} strokeWidth={1.6} style={{ flexShrink: 0, marginTop: 1, color: 'var(--muted-2)' }} />
            <span>For clinical decision support only. Final diagnosis must be made by a certified specialist.</span>
          </div>
        </div>

        {/* RIGHT: Analysis */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <SectionHeader icon={<Cpu size={14} />} label="AI Analysis" step="02" />

          {error && (
            <div className="card" style={{ padding: 16, background: 'var(--risk-bg)', borderColor: 'rgba(180,35,24,0.2)', display: 'flex', gap: 12 }}>
              <AlertTriangle size={18} style={{ color: 'var(--risk-ink)', flexShrink: 0 }} />
              <div>
                <div style={{ fontWeight: 700, color: 'var(--risk-ink)', fontSize: 13 }}>Analysis Failed</div>
                <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 2 }}>{error}</div>
              </div>
            </div>
          )}

          {!result && !loading && <EmptyAnalysis />}
          {loading && <LoadingAnalysis step={analysisStep} />}

          {result && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <EvidenceBanners result={result} />

              {/* Top diagnosis hero */}
              <TopDiagnosisCard result={result} />

              {/* Full prediction breakdown */}
              <DiagnosisResults
                predictions={result.predictions}
                entropy={result.entropy}
                referToSpecialist={result.refer_to_specialist}
              />

              <AttentionHeatmap
                originalPreviewUrl={previewUrl}
                heatmapB64={result.heatmap_b64}
                topClassName={result.top_class_name}
              />

              <ReferenceGallery
                referenceImages={result.reference_images}
                topClassName={result.top_class_name}
              />

              <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', alignItems: 'center', marginTop: 6 }}>
                {result.case_display_id && (
                  <span className="mono" style={{ fontSize: 11, color: 'var(--muted)', marginRight: 'auto' }}>
                    Saved as case #{result.case_display_id}
                  </span>
                )}
                <button className="btn btn-ghost" onClick={handleReset}>
                  <Refresh size={14} /> New Scan
                </button>
                <a className="btn btn-primary" href={caseReportUrl(result.case_id)} download style={{ textDecoration: 'none' }}>
                  <Download size={14} /> Export Clinical Report (PDF)
                </a>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Must match HAM10000 `localization` values (lowercase) so the backend prior keys align.
const ANATOMICAL_SITES = [
  'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand',
  'lower extremity', 'neck', 'scalp', 'trunk', 'upper extremity',
]

const inputStyle = {
  width: '100%', padding: '10px 12px', borderRadius: 10,
  border: '1px solid var(--border)', background: 'var(--surface)',
  fontSize: 13, color: 'var(--ink)', outline: 'none',
}

function FormField({ label, children }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--muted)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>{label}</span>
      {children}
    </label>
  )
}

function StepFlow({ active = 1 }) {
  const steps = [
    { n: 1, label: 'Upload Scan',       Icon: Upload },
    { n: 2, label: 'AI Analysis',       Icon: Cpu },
    { n: 3, label: 'Review & Confirm',  Icon: CheckCircle },
    { n: 4, label: 'Report & Refer',    Icon: ArrowUpRight },
  ]
  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center', background: 'var(--surface)', padding: 8, borderRadius: 14, border: '1px solid var(--border)' }}>
      {steps.map((s, i) => {
        const done = active > s.n
        const cur  = active === s.n
        return (
          <div key={s.n} style={{ display: 'flex', alignItems: 'center', flex: 1, gap: 4 }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 10, padding: '8px 14px', borderRadius: 10, flex: 1,
              background: cur ? 'var(--accent-soft)' : 'transparent',
              color: cur ? 'var(--primary)' : done ? 'var(--ok-ink)' : 'var(--muted)',
              transition: 'all .2s',
            }}>
              <div style={{
                width: 24, height: 24, borderRadius: 999, display: 'grid', placeItems: 'center',
                background: cur ? 'var(--primary)' : done ? 'var(--ok-ink)' : 'var(--bg-2)',
                color: cur || done ? '#fff' : 'var(--muted)',
                fontSize: 11, fontWeight: 700,
              }}>
                {done ? <Check size={13} /> : s.n}
              </div>
              <span style={{ fontSize: 12, fontWeight: 600 }}>{s.label}</span>
            </div>
            {i < steps.length - 1 && <div style={{ width: 16, height: 1, background: 'var(--border)', flexShrink: 0 }} />}
          </div>
        )
      })}
    </div>
  )
}

function EmptyAnalysis() {
  return (
    <div style={{ border: '1px dashed var(--border-strong)', borderRadius: 16, padding: '80px 20px', textAlign: 'center', background: 'var(--surface)' }}>
      <div style={{ width: 64, height: 64, borderRadius: 16, background: 'var(--bg-2)', display: 'grid', placeItems: 'center', margin: '0 auto 16px', color: 'var(--primary)' }}>
        <Microscope size={28} strokeWidth={1.5} />
      </div>
      <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--ink)', marginBottom: 4 }}>No scan loaded</div>
      <div style={{ fontSize: 13, color: 'var(--muted)' }}>Upload an image and run analysis to begin</div>
    </div>
  )
}

function LoadingAnalysis({ step }) {
  const stages = ['Preprocessing', 'Feature Extraction', 'Embedding', 'Prototype Matching', 'Attention Rollout', 'Calibration']
  const idx = Math.max(0, ANALYSIS_STEPS.indexOf(step))
  return (
    <div className="card" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <span className="pulse-dot" style={{ width: 8, height: 8, borderRadius: 999, background: 'var(--primary)' }} />
        <span className="mono" style={{ fontSize: 12, color: 'var(--primary)' }}>{step || 'Initializing…'}</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {stages.map((s, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 22, height: 22, borderRadius: 999, display: 'grid', placeItems: 'center',
              background: i <= idx ? 'var(--primary)' : 'var(--bg-2)',
              color: i <= idx ? '#fff' : 'var(--muted)',
              fontSize: 10, fontWeight: 700, transition: 'background .3s',
            }}>
              {i < idx ? <Check size={11} /> : i + 1}
            </div>
            <span style={{ fontSize: 13, color: i <= idx ? 'var(--ink)' : 'var(--muted)', fontWeight: i === idx ? 600 : 500 }}>{s}</span>
            {i === idx && <div className="shimmer" style={{ flex: 1, height: 6, borderRadius: 99 }} />}
          </div>
        ))}
      </div>
    </div>
  )
}

function TopDiagnosisCard({ result }) {
  const top = result.predictions?.[0]
  if (!top) return null
  const confidence = top.confidence ?? top.probability ?? 0

  return (
    <div className="card fade-up" style={{ padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 18 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>
            Top Differential
          </div>
          <h2 style={{ fontSize: 26, fontWeight: 700, color: 'var(--ink)', letterSpacing: '-0.015em' }}>
            {top.class_name || result.top_class_name}
          </h2>
        </div>
        <RiskPill kind={result.refer_to_specialist ? 'high' : 'moderate'} />
      </div>

      <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
        <ConfidenceRing value={confidence} />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <Stat label="Entropy" value={(result.entropy ?? 0).toFixed(2)} sub="Uncertainty measure" />
          <Stat
            label="Calibration ECE"
            value={result.calibration_ece != null ? result.calibration_ece.toFixed(3) : '—'}
            sub={result.calibration_ece != null
              ? (result.calibration_ece < 0.10 ? 'Well-calibrated (<0.10)' : 'Temperature-scaled')
              : 'n/a'}
          />
          <Stat
            label="Recommendation"
            value={result.refer_to_specialist ? 'Refer to specialist' : 'Routine follow-up'}
            sub={result.refer_to_specialist ? 'Dermatology / Oncology' : '6-month review'}
            highlight={result.refer_to_specialist}
          />
        </div>
      </div>
    </div>
  )
}

function EvidenceBanners({ result }) {
  const meta = result.metadata_used || []
  const banners = []

  if (result.is_unknown) {
    banners.push(
      <Banner key="ood" kind="risk" icon={<AlertTriangle size={14} />}
        title="Out-of-distribution — possible unknown condition"
        body="This image does not confidently match any of the 7 known classes. The prediction below is low-trust; manual specialist review is required." />
    )
  }
  if (result.modality_warning) {
    banners.push(
      <Banner key="mod" kind="warn" icon={<Info size={14} />}
        title="Modality mismatch" body={result.modality_warning} />
    )
  }
  if (meta.length) {
    banners.push(
      <Banner key="meta" kind="ok" icon={<Activity size={14} />}
        title={`Patient metadata applied (${meta.join(', ')})`}
        body={result.metadata_changed_ranking
          ? 'Age/sex/site evidence changed the top prediction (class-conditional prior).'
          : 'Age/sex/site evidence refined the ranking via a class-conditional prior.'} />
    )
  }
  if (result.note_used) {
    banners.push(
      <Banner key="note" kind="ok" icon={<Stethoscope size={14} />}
        title="Clinical note matched"
        body={`Note text most consistent with ${result.note_supports} (BiomedCLIP text match).`} />
    )
  }
  if (!banners.length) return null
  return <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>{banners}</div>
}

function Banner({ kind, icon, title, body }) {
  const c = kind === 'risk'
    ? { bg: 'var(--risk-bg)', ink: 'var(--risk-ink)', bd: 'rgba(180,35,24,0.3)' }
    : kind === 'warn'
    ? { bg: 'var(--warn-bg)', ink: 'var(--warn-ink)', bd: 'rgba(146,64,14,0.3)' }
    : { bg: 'var(--ok-bg)', ink: 'var(--ok-ink)', bd: 'rgba(21,128,61,0.25)' }
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: 12, borderRadius: 10,
      background: c.bg, border: `1px solid ${c.bd}` }}>
      <span style={{ color: c.ink, flexShrink: 0, marginTop: 1 }}>{icon}</span>
      <div>
        <p style={{ fontSize: 12, fontWeight: 700, color: c.ink }}>{title}</p>
        <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2, lineHeight: 1.5 }}>{body}</p>
      </div>
    </div>
  )
}

function ConfidenceRing({ value = 0.94 }) {
  const R   = 52
  const C   = 2 * Math.PI * R
  const pct = Math.min(1, Math.max(0, value))
  return (
    <div style={{ position: 'relative', width: 132, height: 132, flexShrink: 0 }}>
      <svg width="132" height="132" viewBox="0 0 132 132">
        <circle cx="66" cy="66" r={R} stroke="var(--bg-2)" strokeWidth="10" fill="none" />
        <circle
          cx="66" cy="66" r={R}
          stroke="url(#confGrad)" strokeWidth="10" fill="none"
          strokeDasharray={`${C * pct} ${C}`} strokeLinecap="round"
          transform="rotate(-90 66 66)"
          style={{ transition: 'stroke-dasharray 1s cubic-bezier(.2,.8,.2,1)' }}
        />
        <defs>
          <linearGradient id="confGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="var(--primary)" />
            <stop offset="100%" stopColor="var(--accent)" />
          </linearGradient>
        </defs>
      </svg>
      <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--ink)', letterSpacing: '-0.02em' }}>
          {(pct * 100).toFixed(1)}%
        </div>
        <div style={{ fontSize: 10, color: 'var(--muted)', letterSpacing: '0.1em', fontWeight: 600, textTransform: 'uppercase' }}>Confidence</div>
      </div>
    </div>
  )
}

function Stat({ label, value, sub, highlight }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px dashed var(--border)' }}>
      <div>
        <div style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase' }}>{label}</div>
        <div style={{ fontSize: 11, color: 'var(--muted-2)', marginTop: 2 }}>{sub}</div>
      </div>
      <div style={{ fontSize: 13, fontWeight: 700, color: highlight ? 'var(--risk-ink)' : 'var(--ink)' }}>{value}</div>
    </div>
  )
}
