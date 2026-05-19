import { CASES, KPIS, ACTIVITY } from '../data'
import { KpiCard, RiskPill, PlaceholderImage, Avatar } from '../components/common'
import { ArrowRight, Upload, CheckCircle, ArrowUpRight, Folder, Sparkles, Eye, Share } from '../icons'

export default function Dashboard({ setRoute }) {
  const today = new Date()
  const dateStr = today.toLocaleDateString('en-GB', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })

  return (
    <div className="page fade-up">
      {/* Welcome */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontWeight: 700, fontSize: 36, letterSpacing: '-0.02em', color: 'var(--ink)', marginBottom: 6 }}>
          Welcome back, <span style={{ color: 'var(--primary)' }}>Dr. Aris</span>
        </h1>
        <p style={{ color: 'var(--muted)', fontSize: 14 }}>{dateStr} · 8 cases need your attention today</p>
      </div>

      {/* KPI strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 36 }}>
        {KPIS.map((k, i) => <KpiCard key={i} {...k} />)}
      </div>

      {/* Recent cases */}
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ fontWeight: 700, fontSize: 22, color: 'var(--ink)' }}>Recent Cases</h2>
        <button
          onClick={() => setRoute('scan')}
          style={{ color: 'var(--primary)', fontSize: 13, fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: 4 }}
        >
          New Scan <ArrowRight size={14} />
        </button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 36 }}>
        {CASES.map((c, i) => (
          <CaseCard key={c.id} c={c} seed={i} onOpen={() => setRoute('scan')} />
        ))}
      </div>

      {/* Lower section */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20 }}>
        <DiagnosticTrend />
        <ActivityFeed />
      </div>
    </div>
  )
}

function CaseCard({ c, seed, onOpen }) {
  return (
    <div
      className="card"
      style={{ display: 'grid', gridTemplateColumns: '180px 1fr', overflow: 'hidden', transition: 'transform .15s, box-shadow .15s' }}
      onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = 'var(--shadow-md)' }}
      onMouseLeave={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'var(--shadow-sm)' }}
    >
      <PlaceholderImage label={c.modality} seed={seed + 2} height="100%" radius={0} style={{ minHeight: 184 }} />
      <div style={{ padding: 18, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
          <div style={{ minWidth: 0 }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--ink)', marginBottom: 2 }}>{c.name}</h3>
            <div className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>#{c.id} · {c.age}{c.sex.toLowerCase()}</div>
          </div>
          <RiskPill kind={c.risk} />
        </div>
        <p style={{
          fontSize: 13, color: 'var(--muted)', marginTop: 10, lineHeight: 1.5,
          display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
        }}>
          {c.summary}
        </p>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 14 }}>
          <div style={{ display: 'flex', gap: 6 }}>
            <IconBtn><Eye size={15} /></IconBtn>
            <IconBtn><Share size={15} /></IconBtn>
          </div>
          <button className="btn btn-primary" onClick={onOpen} style={{ padding: '8px 16px', fontSize: 12 }}>
            Review Case <ArrowRight size={13} />
          </button>
        </div>
      </div>
    </div>
  )
}

function IconBtn({ children, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        width: 32, height: 32, borderRadius: 8,
        display: 'grid', placeItems: 'center',
        color: 'var(--muted)', border: '1px solid var(--border)', background: 'var(--surface)',
      }}
      onMouseEnter={e => { e.currentTarget.style.background = 'var(--bg-2)'; e.currentTarget.style.color = 'var(--primary)' }}
      onMouseLeave={e => { e.currentTarget.style.background = 'var(--surface)'; e.currentTarget.style.color = 'var(--muted)' }}
    >
      {children}
    </button>
  )
}

function DiagnosticTrend() {
  const days = Array.from({ length: 14 }, (_, i) => ({
    scans: 12 + Math.round(Math.sin(i * 0.7) * 6 + (i * 0.3)),
    flags: Math.max(0, Math.round(2 + Math.sin(i * 0.5) * 2 + (i * 0.1))),
  }))
  const max = Math.max(...days.map(d => d.scans))

  return (
    <div className="card" style={{ padding: 22 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700 }}>Diagnostic Activity</h3>
        <div style={{ display: 'flex', gap: 14, fontSize: 12, color: 'var(--muted)' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: 'var(--primary)' }} /> Scans
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: 'var(--accent-soft)' }} /> Flagged
          </span>
        </div>
      </div>
      <p style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 22 }}>Last 14 days</p>

      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, height: 160, paddingBottom: 8, borderBottom: '1px solid var(--border)' }}>
        {days.map((d, i) => (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ width: '100%', display: 'flex', flexDirection: 'column-reverse', height: '100%', gap: 2 }}>
              <div style={{ background: 'var(--primary)', height: `${(d.scans / max) * 80}%`, borderRadius: '4px 4px 0 0' }} />
              <div style={{ background: 'var(--accent-soft)', height: `${(d.flags / max) * 60}%`, borderRadius: '4px 4px 0 0' }} />
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, fontSize: 10, color: 'var(--muted-2)' }} className="mono">
        <span>14d ago</span><span>7d</span><span>Today</span>
      </div>
    </div>
  )
}

function ActivityFeed() {
  const ICONS = {
    upload:  { I: Upload,      c: 'var(--primary)' },
    confirm: { I: CheckCircle, c: 'var(--ok-ink)' },
    refer:   { I: ArrowUpRight,c: 'var(--accent)' },
    archive: { I: Folder,      c: 'var(--muted)' },
    system:  { I: Sparkles,    c: 'var(--primary)' },
  }

  return (
    <div className="card" style={{ padding: 22 }}>
      <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}>Activity</h3>
      <p style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 14 }}>Today</p>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {ACTIVITY.map((a, i) => {
          const Ico = ICONS[a.kind] || ICONS.upload
          return (
            <div key={i} style={{
              display: 'flex', gap: 12, padding: '10px 0',
              borderBottom: i < ACTIVITY.length - 1 ? '1px solid var(--border)' : 'none',
            }}>
              <div style={{
                width: 32, height: 32, borderRadius: 8,
                background: 'var(--bg-2)', display: 'grid', placeItems: 'center',
                color: Ico.c, flexShrink: 0,
              }}>
                <Ico.I size={15} />
              </div>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>{a.action}</div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--muted)', marginTop: 1 }}>{a.meta}</div>
              </div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--muted-2)', flexShrink: 0 }}>{a.time}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
