import { useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { getCase, caseReportUrl, timeAgo } from '../api'
import { RiskPill } from '../components/common'
import DiagnosisResults from '../components/DiagnosisResults'
import { X, Download, AlertTriangle, Activity, Stethoscope } from '../icons'

const riskKind = r => (r === 'high' ? 'high' : r === 'medium' ? 'medium' : 'normal')

export default function CaseDetail({ caseId, onClose }) {
  const [data, setData]   = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getCase(caseId).then(setData).catch(e => setError(e.message))
  }, [caseId])

  useEffect(() => {
    const h = e => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', h)
    return () => document.removeEventListener('keydown', h)
  }, [onClose])

  const result = data?.result

  return createPortal(
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(14,26,54,0.45)',
        backdropFilter: 'blur(4px)', zIndex: 100,
        display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
        padding: '40px 24px', overflowY: 'auto',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        className="fade-up"
        style={{
          background: 'var(--surface)', borderRadius: 20, maxWidth: 880, width: '100%',
          maxHeight: '90vh', overflow: 'auto', boxShadow: 'var(--shadow-lg)', position: 'relative',
        }}
      >
        <button
          onClick={onClose}
          style={{
            position: 'absolute', top: 14, right: 14, width: 36, height: 36, borderRadius: 10,
            background: 'rgba(255,255,255,0.9)', color: 'var(--ink)', zIndex: 2,
            display: 'grid', placeItems: 'center', boxShadow: 'var(--shadow-sm)',
          }}
        >
          <X size={16} />
        </button>

        {error && (
          <div style={{ padding: 48, textAlign: 'center' }}>
            <AlertTriangle size={26} style={{ color: 'var(--warn-ink)', margin: '0 auto 10px' }} />
            <div style={{ fontSize: 14, fontWeight: 700 }}>Could not load case</div>
            <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>{error}</div>
          </div>
        )}

        {!data && !error && (
          <div style={{ padding: 24 }}>
            <div className="shimmer" style={{ height: 220, borderRadius: 14, marginBottom: 16 }} />
            <div className="shimmer" style={{ height: 120, borderRadius: 14 }} />
          </div>
        )}

        {data && (
          <>
            {/* Header: image + summary */}
            <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 0 }}>
              <img
                src={data.thumbnail_url}
                alt="Lesion scan"
                style={{ width: '100%', height: '100%', minHeight: 240, objectFit: 'cover', display: 'block' }}
              />
              <div style={{ padding: '26px 56px 22px 26px' }}>
                <div className="mono" style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>
                  #{data.case_id} · {timeAgo(data.created_at)}
                </div>
                <h2 style={{ fontSize: 24, fontWeight: 700, letterSpacing: '-0.01em', marginBottom: 4 }}>
                  {data.patient_name || 'Unnamed Patient'}
                </h2>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                  <RiskPill kind={riskKind(data.risk)} />
                  <span style={{ fontSize: 12, color: 'var(--muted)' }}>
                    {data.scan_type || 'Dermoscopy'}
                  </span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                  <HeaderFact label="Top Differential" value={data.top_class_name} strong />
                  <HeaderFact label="Confidence" value={`${(data.confidence * 100).toFixed(1)}%`} />
                  <HeaderFact label="Age / Sex" value={`${data.age || '—'} / ${data.sex || '—'}`} />
                  <HeaderFact label="Site" value={data.localization ? data.localization : '—'} />
                </div>
              </div>
            </div>

            <div style={{ padding: '20px 26px 26px', display: 'flex', flexDirection: 'column', gap: 16 }}>
              {result?.is_unknown && (
                <div style={{ display: 'flex', gap: 10, padding: 12, borderRadius: 10, background: 'var(--risk-bg)', border: '1px solid rgba(180,35,24,0.3)' }}>
                  <AlertTriangle size={14} style={{ color: 'var(--risk-ink)', flexShrink: 0, marginTop: 2 }} />
                  <div>
                    <p style={{ fontSize: 12, fontWeight: 700, color: 'var(--risk-ink)' }}>Out-of-distribution — possible unknown condition</p>
                    <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>This prediction is low-trust; manual specialist review required.</p>
                  </div>
                </div>
              )}

              {data.clinical_note && (
                <div style={{ padding: 14, borderRadius: 12, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, color: 'var(--muted)' }}>
                    <Stethoscope size={13} />
                    <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Clinical Note</span>
                  </div>
                  <p style={{ fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.55 }}>{data.clinical_note}</p>
                </div>
              )}

              {result && (
                <DiagnosisResults
                  predictions={result.predictions}
                  entropy={result.entropy}
                  referToSpecialist={result.refer_to_specialist}
                />
              )}

              {result?.heatmap_b64 && (
                <div className="card" style={{ overflow: 'hidden' }}>
                  <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Activity size={13} style={{ color: 'var(--accent)' }} />
                    <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--muted)' }}>
                      Attention Rollout
                    </span>
                  </div>
                  <div style={{ padding: 16 }}>
                    <img
                      src={`data:image/png;base64,${result.heatmap_b64}`}
                      alt="Attention heatmap"
                      style={{ maxWidth: 360, width: '100%', borderRadius: 12, border: '1px solid var(--border)', display: 'block' }}
                    />
                  </div>
                </div>
              )}

              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <a className="btn btn-primary" href={caseReportUrl(data.id)} download
                  style={{ textDecoration: 'none' }}>
                  <Download size={14} /> Download Clinical Report (PDF)
                </a>
              </div>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body,
  )
}

function HeaderFact({ label, value, strong }) {
  return (
    <div style={{ padding: '9px 12px', background: 'var(--bg)', borderRadius: 10 }}>
      <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 3 }}>
        {label}
      </div>
      <div style={{ fontSize: 13, fontWeight: strong ? 700 : 600, color: strong ? 'var(--primary)' : 'var(--ink)', textTransform: label === 'Site' ? 'capitalize' : 'none' }}>
        {value}
      </div>
    </div>
  )
}
