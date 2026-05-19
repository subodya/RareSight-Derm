import { useState } from 'react'
import { Eye, Info } from 'lucide-react'

export default function AttentionHeatmap({ originalPreviewUrl, heatmapB64, topClassName }) {
  const [tooltip, setTooltip] = useState(false)

  if (!heatmapB64 && !originalPreviewUrl) return null

  return (
    <div className="card" style={{ overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Eye size={14} style={{ color: 'var(--accent)' }} />
          <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--muted)' }}>
            AI Attention Rollout
          </span>
          <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 999, background: 'var(--accent-soft)', color: 'var(--primary)', fontFamily: "'JetBrains Mono',monospace", fontWeight: 700 }}>
            XAI
          </span>
        </div>
        <div style={{ position: 'relative' }}>
          <button
            style={{ width: 28, height: 28, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)' }}
            onMouseEnter={() => setTooltip(true)}
            onMouseLeave={() => setTooltip(false)}
          >
            <Info size={13} />
          </button>
          {tooltip && (
            <div style={{
              position: 'absolute', right: 0, top: 34, width: 280, padding: 14, borderRadius: 12, zIndex: 20,
              background: 'var(--ink)', border: '1px solid var(--ink-2)', color: '#fff', fontSize: 12, lineHeight: 1.5,
            }}>
              <p style={{ fontWeight: 700, marginBottom: 6 }}>About Attention Rollout</p>
              <p>Tracks attention flow through all transformer blocks of BiomedCLIP (ViT-B/16). Each patch's importance is accumulated layer-by-layer to show which regions the model attended to most.</p>
              <p style={{ marginTop: 8, fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: 'var(--muted-2)' }}>Method: Abnar & Zuidema (2020)</p>
            </div>
          )}
        </div>
      </div>

      {/* Diagnosis label */}
      <div style={{ padding: '14px 20px 0' }}>
        <p style={{ fontSize: 12, color: 'var(--muted)' }}>
          Attention regions for: <span style={{ fontWeight: 700, color: 'var(--ink)' }}>{topClassName}</span>
        </p>
      </div>

      {/* Image pair */}
      <div style={{ padding: 20, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <ImageCard src={originalPreviewUrl} label="Patient Scan"      tag="ORIGINAL" tagColor="var(--accent)" />
        <ImageCard src={heatmapB64 ? `data:image/png;base64,${heatmapB64}` : null}
          label="Attention Heatmap" tag="XAI" tagColor="var(--warn-ink)" placeholder={!heatmapB64} />
      </div>

      {/* Legend */}
      <div style={{ padding: '0 20px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: 10, borderRadius: 10, background: 'var(--bg)' }}>
          <div style={{ display: 'flex', gap: 4 }}>
            {['#0000ff','#00ff00','#ffff00','#ff4400','#ff0000'].map((c, i) => (
              <div key={i} style={{ width: 20, height: 8, borderRadius: 3, background: c, opacity: 0.8 }} />
            ))}
          </div>
          <span style={{ fontSize: 11, color: 'var(--muted)' }}>
            Low → High attention · Red = highest model focus
          </span>
        </div>
      </div>
    </div>
  )
}

function ImageCard({ src, label, tag, tagColor, placeholder }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{
        borderRadius: 12, overflow: 'hidden', position: 'relative',
        background: 'var(--bg-2)', border: '1px solid var(--border)', aspectRatio: '1',
      }}>
        {placeholder || !src ? (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <p className="mono" style={{ fontSize: 11, color: 'var(--muted-2)' }}>UNAVAILABLE</p>
          </div>
        ) : (
          <img src={src} alt={label} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        )}
        <div style={{ position: 'absolute', top: 8, left: 8 }}>
          <span style={{
            fontSize: 10, fontFamily: "'JetBrains Mono',monospace", fontWeight: 700,
            padding: '2px 8px', borderRadius: 999,
            background: 'rgba(255,255,255,0.9)', color: tagColor,
          }}>
            {tag}
          </span>
        </div>
      </div>
      <p style={{ fontSize: 12, textAlign: 'center', color: 'var(--muted)', fontWeight: 500 }}>{label}</p>
    </div>
  )
}
