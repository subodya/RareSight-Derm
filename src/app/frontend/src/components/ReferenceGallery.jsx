import { BookOpen } from 'lucide-react'

export default function ReferenceGallery({ referenceImages, topClassName }) {
  if (!referenceImages?.length) return null

  return (
    <div className="card" style={{ overflow: 'hidden' }}>
      <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <BookOpen size={14} style={{ color: 'var(--accent)' }} />
          <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--muted)' }}>
            Reference Cases: <span style={{ color: 'var(--ink)', textTransform: 'none', fontWeight: 700 }}>{topClassName}</span>
          </span>
        </div>
        <p style={{ fontSize: 12, color: 'var(--muted)' }}>
          Textbook examples sharing visual similarity with the patient scan.
        </p>
      </div>

      <div style={{ padding: 20 }}>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {referenceImages.slice(0, 3).map((b64, i) => (
            <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
              <div style={{
                width: 110, height: 110, borderRadius: 12, overflow: 'hidden', position: 'relative',
                border: '1px solid var(--border)', background: 'var(--bg-2)',
              }}>
                <img
                  src={`data:image/jpeg;base64,${b64}`}
                  alt={`Reference ${i + 1}`}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
                <div style={{ position: 'absolute', top: 4, left: 4 }}>
                  <span style={{
                    fontSize: 9, fontFamily: "'JetBrains Mono',monospace", fontWeight: 700,
                    padding: '2px 6px', borderRadius: 4,
                    background: 'var(--accent-soft)', color: 'var(--primary)',
                  }}>
                    REF {i + 1}
                  </span>
                </div>
              </div>
              <p className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>Reference {i + 1}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
