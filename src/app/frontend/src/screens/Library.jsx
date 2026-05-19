import { useState, useEffect } from 'react'
import { Search, Filter, Upload, ArrowRight, X, Dna, Calendar, Globe, Database, Crosshair } from '../icons'
import { CategoryPill, PlaceholderImage, KpiCard } from '../components/common'
import { DISEASES, LIBRARY_KPIS } from '../data'
import ContributeModal from './ContributeModal'

export default function Library() {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('All')
  const [selected, setSelected] = useState(null)
  const [contributing, setContributing] = useState(false)

  const categories = ['All', 'Malignant', 'Benign', 'Precancerous']

  const filtered = DISEASES.filter(d => {
    const q = query.toLowerCase()
    const matchQuery = q === '' ||
      d.name.toLowerCase().includes(q) ||
      d.code.toLowerCase().includes(q) ||
      d.summary.toLowerCase().includes(q)
    const matchFilter = filter === 'All' || d.category === filter
    return matchQuery && matchFilter
  })

  return (
    <div className="page fade-up">
      {/* KPI strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 28 }}>
        {LIBRARY_KPIS.map((k, i) => <KpiCard key={i} {...k} />)}
      </div>

      {/* Search + filters */}
      <div className="card" style={{ padding: 18, marginBottom: 24 }}>
        <div style={{ position: 'relative', marginBottom: 14 }}>
          <Search size={16} style={{ position: 'absolute', left: 16, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted)' }} />
          <input
            type="text"
            placeholder="Search by disease name, ICD-10 code, morphology, or clinical sign…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            className="input"
            style={{ padding: '12px 14px 12px 44px' }}
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', marginRight: 4 }}>
            Quick Filters
          </span>
          {categories.map(cat => (
            <button
              key={cat}
              onClick={() => setFilter(cat)}
              style={{
                padding: '7px 16px', borderRadius: 999,
                background: filter === cat ? 'var(--primary)' : 'var(--surface)',
                color: filter === cat ? '#fff' : 'var(--ink-2)',
                border: filter === cat ? '1px solid var(--primary)' : '1px solid var(--border)',
                fontSize: 12, fontWeight: 600, transition: 'all .15s',
              }}
            >
              {cat}
            </button>
          ))}
          <div style={{ width: 1, height: 22, background: 'var(--border)', margin: '0 6px' }} />
          <button className="btn btn-ghost" style={{ padding: '7px 14px', fontSize: 12 }}>
            <Filter size={13} /> Advanced Filters
          </button>
          <div style={{ flex: 1 }} />
          <span style={{ fontSize: 12, color: 'var(--muted)' }}>{filtered.length} of {DISEASES.length} diseases</span>
        </div>
      </div>

      {/* Cards grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        {filtered.map((d, i) => (
          <DiseaseCard key={d.name} d={d} seed={i + 1} onOpen={() => setSelected(d)} />
        ))}
        <ContributeCard onClick={() => setContributing(true)} />
      </div>

      {selected && <DiseaseModal disease={selected} onClose={() => setSelected(null)} />}
      {contributing && <ContributeModal onClose={() => setContributing(false)} />}
    </div>
  )
}

function DiseaseCard({ d, seed, onOpen }) {
  return (
    <div
      className="card"
      style={{ overflow: 'hidden', cursor: 'pointer', transition: 'transform .15s, box-shadow .15s' }}
      onClick={onOpen}
      onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = 'var(--shadow-md)' }}
      onMouseLeave={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'var(--shadow-sm)' }}
    >
      <div style={{ position: 'relative' }}>
        <PlaceholderImage label={`${d.modality} · REFERENCE`} height={190} radius={0} seed={seed * 3} />
        <div style={{ position: 'absolute', top: 12, left: 12 }}>
          <CategoryPill category={d.category} />
        </div>
      </div>
      <div style={{ padding: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
          <h3 style={{ fontSize: 15, fontWeight: 700, color: 'var(--ink)', lineHeight: 1.3 }}>{d.name}</h3>
          <span className="mono" style={{ fontSize: 11, color: 'var(--muted)', flexShrink: 0, marginLeft: 8 }}>{d.code}</span>
        </div>
        <p style={{
          fontSize: 12, color: 'var(--muted)', lineHeight: 1.5, marginBottom: 14,
          display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
        }}>{d.summary}</p>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 12, borderTop: '1px solid var(--border)' }}>
          <span style={{ fontSize: 11, color: 'var(--muted)' }}>
            <span className="mono" style={{ color: 'var(--ink)', fontWeight: 600 }}>{d.cases}</span> cases
          </span>
          <button style={{ width: 30, height: 30, borderRadius: 8, background: 'var(--bg-2)', display: 'grid', placeItems: 'center', color: 'var(--primary)' }}>
            <ArrowRight size={14} />
          </button>
        </div>
      </div>
    </div>
  )
}

function ContributeCard({ onClick }) {
  return (
    <div className="card" style={{
      background: 'linear-gradient(155deg, var(--primary) 0%, #2A4FAE 100%)',
      border: 'none', color: '#fff', padding: 24,
      display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: 380,
    }}>
      <div style={{ width: 48, height: 48, borderRadius: 12, background: 'rgba(255,255,255,0.12)', display: 'grid', placeItems: 'center', marginBottom: 18 }}>
        <Upload size={22} strokeWidth={1.6} />
      </div>
      <div style={{ flex: 1 }}>
        <h3 style={{ fontSize: 18, fontWeight: 700, color: '#fff', marginBottom: 8 }}>Contribute Data</h3>
        <p style={{ fontSize: 13, color: 'rgba(255,255,255,0.78)', lineHeight: 1.5, marginBottom: 18 }}>
          Help expand the global RareSight knowledge base. Anonymized scans accepted from verified clinicians.
        </p>
      </div>
      <button onClick={onClick} style={{
        background: '#fff', color: 'var(--primary)', padding: '10px 16px',
        borderRadius: 10, fontSize: 13, fontWeight: 700,
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6,
      }}>
        Submit a Disease <ArrowRight size={14} />
      </button>
    </div>
  )
}

function DiseaseModal({ disease, onClose }) {
  useEffect(() => {
    const h = e => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', h)
    return () => document.removeEventListener('keydown', h)
  }, [onClose])

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(14,26,54,0.45)',
        backdropFilter: 'blur(4px)', zIndex: 100, display: 'grid', placeItems: 'center', padding: 24,
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        className="fade-up"
        style={{
          background: 'var(--surface)', borderRadius: 20, maxWidth: 880, width: '100%',
          maxHeight: '90vh', overflow: 'auto', boxShadow: 'var(--shadow-lg)',
        }}
      >
        <div style={{ position: 'relative' }}>
          <PlaceholderImage label={`${disease.modality} · TYPICAL PRESENTATION`} height={240} radius={0} seed={4} />
          <button
            onClick={onClose}
            style={{
              position: 'absolute', top: 14, right: 14, width: 36, height: 36, borderRadius: 10,
              background: 'rgba(255,255,255,0.9)', color: 'var(--ink)',
              display: 'grid', placeItems: 'center', boxShadow: 'var(--shadow-sm)',
            }}
          >
            <X size={16} />
          </button>
          <div style={{ position: 'absolute', top: 14, left: 14 }}>
            <CategoryPill category={disease.category} />
          </div>
        </div>

        <div style={{ padding: '28px 32px 32px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 14 }}>
            <div>
              <h2 style={{ fontSize: 26, fontWeight: 700, letterSpacing: '-0.01em', marginBottom: 4 }}>{disease.name}</h2>
              <div className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>ICD-10 · {disease.code}</div>
            </div>
            <button className="btn btn-primary"><Crosshair size={14} /> Compare to Case</button>
          </div>

          <p style={{ marginTop: 16, fontSize: 14, color: 'var(--ink-2)', lineHeight: 1.6 }}>{disease.summary}</p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 22 }}>
            <Fact icon={<Dna size={16} />} label="Risk Factors" value={disease.riskFactors} />
            <Fact icon={<Calendar size={16} />} label="Typical Onset" value={disease.age} />
            <Fact icon={<Globe size={16} />} label="Prevalence" value={disease.prevalence} />
            <Fact icon={<Database size={16} />} label="Documented" value={disease.cases} />
          </div>

          <h3 style={{ fontSize: 14, fontWeight: 700, marginTop: 26, marginBottom: 12 }}>Reference Cases</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8 }}>
            {[0,1,2,3,4].map(i => (
              <PlaceholderImage key={i} label="" height={100} seed={i + 13} radius={8} />
            ))}
          </div>

          <h3 style={{ fontSize: 14, fontWeight: 700, marginTop: 22, marginBottom: 10 }}>Diagnostic Criteria</h3>
          <ul style={{ paddingLeft: 18, fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.7 }}>
            {(disease.criteria || [
              'Visual inspection of lesion morphology and border characteristics',
              'Dermoscopic pattern analysis (ABCDE criteria)',
              'Histopathological confirmation for definitive diagnosis',
              'Clinical correlation with patient history and risk factors',
            ]).map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      </div>
    </div>
  )
}

function Fact({ icon, label, value }) {
  return (
    <div style={{ padding: '12px 14px', background: 'var(--bg)', borderRadius: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--muted)', marginBottom: 6 }}>
        {icon}
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>{label}</span>
      </div>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>{value}</div>
    </div>
  )
}
