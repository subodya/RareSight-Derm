import { useEffect, useRef } from 'react'
import { AlertTriangle, ShieldAlert, TrendingUp, Activity } from 'lucide-react'

function getProbColor(prob) {
  if (prob >= 0.7) return 'var(--ok-ink)'
  if (prob >= 0.4) return 'var(--warn-ink)'
  return 'var(--risk-ink)'
}

function getRiskStyle(risk) {
  if (risk === 'high')   return { bg: 'var(--risk-bg)', text: 'var(--risk-ink)', border: 'rgba(180,35,24,0.3)' }
  if (risk === 'medium') return { bg: 'var(--warn-bg)', text: 'var(--warn-ink)', border: 'rgba(146,64,14,0.3)' }
  return { bg: 'var(--ok-bg)', text: 'var(--ok-ink)', border: 'rgba(21,128,61,0.25)' }
}

function getConfidenceLevel(entropy) {
  if (entropy == null) return null
  const maxEntropy = Math.log(7)
  const ratio = entropy / maxEntropy
  if (ratio < 0.35) return { label: 'HIGH',     color: 'var(--ok-ink)',   desc: 'Strong match detected' }
  if (ratio < 0.65) return { label: 'MODERATE', color: 'var(--warn-ink)', desc: 'Moderate certainty' }
  return              { label: 'LOW',      color: 'var(--risk-ink)', desc: 'High uncertainty — consider referral' }
}

function AnimatedBar({ prob, color }) {
  const barRef = useRef(null)
  useEffect(() => {
    if (!barRef.current) return
    barRef.current.style.width = '0%'
    const t = setTimeout(() => {
      if (barRef.current) barRef.current.style.width = `${prob * 100}%`
    }, 80)
    return () => clearTimeout(t)
  }, [prob])

  return (
    <div className="bar-track">
      <div
        ref={barRef}
        style={{
          height: '100%',
          borderRadius: 999,
          background: `linear-gradient(90deg, var(--primary), var(--accent))`,
          transition: 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
      />
    </div>
  )
}

export default function DiagnosisResults({ predictions, entropy, referToSpecialist }) {
  if (!predictions?.length) return null

  const confidence = getConfidenceLevel(entropy)
  const hasMelanoma = predictions.some(
    p => p.class_name?.toLowerCase().includes('melanoma') && p.probability > 0.30
  )

  return (
    <div className="card" style={{ overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '18px 20px 16px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <TrendingUp size={14} style={{ color: 'var(--accent)' }} />
            <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--muted)' }}>
              Differential Diagnosis
            </span>
          </div>
          {confidence && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 12, color: 'var(--muted)' }}>Confidence:</span>
              <span className="mono" style={{ fontSize: 12, fontWeight: 700, color: confidence.color }}>{confidence.label}</span>
              <div className="pulse-dot" style={{ width: 6, height: 6, borderRadius: 999, background: confidence.color }} />
            </div>
          )}
        </div>
        {confidence && <p style={{ fontSize: 12, marginTop: 4, color: 'var(--muted)' }}>{confidence.desc}</p>}
      </div>

      {/* Refer to specialist banner */}
      {referToSpecialist && (
        <div style={{ margin: '16px 20px 0', display: 'flex', alignItems: 'center', gap: 10, padding: 12, borderRadius: 10, background: 'var(--risk-bg)', border: '1px solid rgba(180,35,24,0.3)' }}>
          <ShieldAlert size={14} style={{ color: 'var(--risk-ink)', flexShrink: 0 }} />
          <div>
            <p style={{ fontSize: 12, fontWeight: 700, color: 'var(--risk-ink)' }}>Specialist Referral Recommended</p>
            <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>High prediction uncertainty — telehealth consult advised</p>
          </div>
        </div>
      )}

      {/* Melanoma alert */}
      {hasMelanoma && (
        <div style={{ margin: '12px 20px 0', display: 'flex', alignItems: 'center', gap: 10, padding: 12, borderRadius: 10, background: 'var(--risk-bg)', border: '1px solid rgba(180,35,24,0.4)' }}>
          <AlertTriangle size={14} style={{ color: 'var(--risk-ink)', flexShrink: 0 }} />
          <div>
            <p style={{ fontSize: 12, fontWeight: 700, color: 'var(--risk-ink)' }}>HIGH RISK — Melanoma features detected</p>
            <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>Immediate dermatologist referral is strongly recommended</p>
          </div>
        </div>
      )}

      {/* Prediction rows */}
      <div style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
        {predictions.map((pred, i) => {
          const pct   = ((pred.probability ?? pred.confidence ?? 0) * 100).toFixed(1)
          const color = getProbColor(pred.probability ?? pred.confidence ?? 0)
          const risk  = getRiskStyle(pred.risk)

          return (
            <div key={pred.class_id || i}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <span style={{
                  width: 26, height: 26, borderRadius: 8,
                  fontSize: 11, fontFamily: "'JetBrains Mono',monospace", fontWeight: 700,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                  background: 'var(--accent-soft)', color: 'var(--primary)',
                }}>
                  #{i + 1}
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>{pred.class_name}</span>
                    <span style={{
                      fontSize: 10, padding: '2px 8px', borderRadius: 999, fontWeight: 700,
                      background: risk.bg, color: risk.text, border: `1px solid ${risk.border}`,
                    }}>
                      {(pred.risk || 'low').toUpperCase()}
                    </span>
                  </div>
                </div>
                <span className="mono" style={{ fontSize: 13, fontWeight: 700, color, flexShrink: 0 }}>{pct}%</span>
              </div>
              <AnimatedBar prob={pred.probability ?? pred.confidence ?? 0} color={color} />
            </div>
          )
        })}
      </div>

      {/* Entropy meter */}
      {entropy != null && (
        <div style={{ padding: '0 20px 20px' }}>
          <div style={{ padding: 12, borderRadius: 10, background: 'var(--bg)', border: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Activity size={11} style={{ color: 'var(--muted)' }} />
                <span className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>PREDICTION ENTROPY</span>
              </div>
              <span className="mono" style={{ fontSize: 11, color: 'var(--muted-2)' }}>{entropy.toFixed(3)} nats</span>
            </div>
            <div style={{ height: 4, borderRadius: 999, background: 'var(--bg-2)', overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 999,
                width: `${Math.min((entropy / Math.log(7)) * 100, 100)}%`,
                background: 'linear-gradient(90deg, var(--ok-ink), var(--warn-ink), var(--risk-ink))',
                transition: 'width 1s',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
              <span className="mono" style={{ fontSize: 10, color: 'var(--muted-2)' }}>CERTAIN</span>
              <span className="mono" style={{ fontSize: 10, color: 'var(--muted-2)' }}>UNCERTAIN</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
