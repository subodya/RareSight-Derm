import { Grid, Microscope, Book, User, Bell, Database } from '../icons'
import { PROFILE } from '../data'

// ── Logo ────────────────────────────────────────────────────────
export function Logo() {
  return (
    <img
      src="/raresight-logo.png"
      alt="Raresight"
      draggable={false}
      style={{
        height: '88px',
        padding: '10.4px 0px 0px',
        width: '237px',
        margin: '9.6px 0px 0px -20px',
        objectFit: 'contain',
        display: 'block',
        userSelect: 'none',
      }}
    />
  )
}

// ── Sidebar ─────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', Icon: Grid },
  { id: 'scan',      label: 'Scan',      Icon: Microscope },
  { id: 'library',   label: 'Library',   Icon: Book },
  { id: 'profile',   label: 'Profile',   Icon: User },
]

export function Sidebar({ route, setRoute }) {
  return (
    <aside style={{
      background: 'var(--bg)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      gap: 24,
      fontFamily: '"JetBrains Mono"',
      padding: '22.4px 18px 18px',
    }}>
      <div style={{ paddingLeft: 6, textAlign: 'center' }}>
        <Logo />
      </div>

      <nav style={{
        display: 'flex',
        flexDirection: 'column',
        marginTop: 8,
        fontFamily: 'Manrope',
        gap: 9,
        fontSize: 15,
      }}>
        {NAV_ITEMS.map(({ id, label, Icon: IconC }) => {
          const active = route === id
          return (
            <button
              key={id}
              onClick={() => setRoute(id)}
              style={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'center',
                gap: 12,
                padding: '11px 14px',
                borderRadius: 12,
                background: active ? 'var(--accent-soft)' : 'transparent',
                color: active ? 'var(--primary)' : 'var(--muted)',
                fontWeight: active ? 600 : 500,
                fontSize: 14,
                textAlign: 'left',
                width: '100%',
                transition: 'background .15s, color .15s',
              }}
              onMouseEnter={e => { if (!active) e.currentTarget.style.background = 'var(--bg-2)' }}
              onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent' }}
            >
              <IconC size={18} strokeWidth={active ? 1.8 : 1.6} />
              <span>{label}</span>
            </button>
          )
        })}
      </nav>

      <div style={{ marginTop: 'auto' }}>
        {/* Cloud storage widget */}
        <div style={{ padding: '14px', borderRadius: 12, background: 'var(--bg-2)', marginBottom: 14 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--ink-2)' }}>Cloud Storage</span>
            <Database size={14} style={{ color: 'var(--muted)' }} />
          </div>
          <div className="bar-track" style={{ height: 6, background: '#fff' }}>
            <div className="bar-fill" style={{ width: '75%' }} />
          </div>
          <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 6 }} className="mono">
            7.5 / 10 GB used
          </div>
        </div>

        {/* User card */}
        <button
          onClick={() => setRoute('profile')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: 10,
            borderRadius: 12,
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            width: '100%',
            textAlign: 'left',
          }}
        >
          <Avatar name={PROFILE.name} size={36} />
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>{PROFILE.name}</div>
            <div style={{ fontSize: 11, color: 'var(--muted)' }}>General Practitioner</div>
          </div>
        </button>
      </div>
    </aside>
  )
}

// ── TopBar ───────────────────────────────────────────────────────
export function TopBar({ title, primaryAction }) {
  return (
    <header className="topbar">
      <h1 style={{ letterSpacing: '-0.015em', fontWeight: 800, color: 'rgb(30,58,138)', fontSize: 30 }}>
        {title}
      </h1>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        {primaryAction}
        <button
          title="Notifications"
          style={{
            width: 40, height: 40, borderRadius: 10,
            display: 'grid', placeItems: 'center',
            color: 'var(--ink-2)', position: 'relative',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = 'var(--bg-2)' }}
          onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
        >
          <Bell size={18} />
          <span style={{
            position: 'absolute', top: 9, right: 10,
            width: 8, height: 8, borderRadius: 999,
            background: '#EF4444',
            border: '2px solid var(--bg)',
          }} />
        </button>
      </div>
    </header>
  )
}

// ── Avatar ───────────────────────────────────────────────────────
export function Avatar({ name = 'Dr. Aris', size = 32, color }) {
  const initials = name
    .split(' ')
    .map(n => n[0])
    .slice(0, 2)
    .join('')
    .replace('.', '')
    .toUpperCase()
  return (
    <div style={{
      width: size, height: size, borderRadius: 999,
      background: color || 'var(--primary)',
      color: '#fff',
      display: 'grid', placeItems: 'center',
      fontSize: size * 0.36,
      fontWeight: 700,
      flexShrink: 0,
      letterSpacing: '0.02em',
    }}>
      {initials}
    </div>
  )
}

// ── RiskPill ─────────────────────────────────────────────────────
export function RiskPill({ kind }) {
  const styles = {
    high:        { bg: 'var(--risk-bg)', ink: 'var(--risk-ink)', label: 'High Risk' },
    observation: { bg: 'var(--obs-bg)',  ink: 'var(--obs-ink)',  label: 'Observation' },
    normal:      { bg: 'var(--ok-bg)',   ink: 'var(--ok-ink)',   label: 'Normal' },
    urgent:      { bg: 'var(--risk-bg)', ink: 'var(--risk-ink)', label: 'Urgent' },
    medium:      { bg: 'var(--warn-bg)', ink: 'var(--warn-ink)', label: 'Medium Risk' },
  }
  const s = styles[kind] || styles.observation
  return <span className="pill" style={{ background: s.bg, color: s.ink }}>{s.label}</span>
}

// ── CategoryPill ─────────────────────────────────────────────────
export function CategoryPill({ category }) {
  const map = {
    Malignant:    { bg: 'var(--risk-bg)',  ink: 'var(--risk-ink)' },
    Benign:       { bg: 'var(--ok-bg)',    ink: 'var(--ok-ink)' },
    Precancerous: { bg: 'var(--warn-bg)',  ink: 'var(--warn-ink)' },
    Rare:         { bg: 'var(--obs-bg)',   ink: 'var(--obs-ink)' },
    Observation:  { bg: 'var(--accent-soft)', ink: 'var(--primary)' },
  }
  const s = map[category] || map.Observation
  return <span className="pill" style={{ background: s.bg, color: s.ink }}>{category}</span>
}

// ── PlaceholderImage ─────────────────────────────────────────────
export function PlaceholderImage({ label = 'DERMOSCOPY · REF', corner = null, height = 180, radius = 12, style: extraStyle = {}, seed = 0 }) {
  const blobs = [
    { x: 30 + seed * 13 % 50, y: 30 + seed * 7 % 40, c: 'rgba(30,58,138,0.18)' },
    { x: 60 + seed * 11 % 30, y: 60 + seed * 5 % 30, c: 'rgba(37,99,235,0.12)' },
    { x: 80 - seed * 9  % 30, y: 20 + seed * 3 % 20, c: 'rgba(94,234,212,0.10)' },
  ]
  return (
    <div
      className="ph-img"
      style={{
        height,
        borderRadius: radius,
        backgroundImage: `
          repeating-linear-gradient(135deg, rgba(30,58,138,0.035) 0px, rgba(30,58,138,0.035) 1px, transparent 1px, transparent 14px),
          ${blobs.map(b => `radial-gradient(circle at ${b.x}% ${b.y}%, ${b.c}, transparent 55%)`).join(', ')},
          linear-gradient(135deg, #F0F4FB, #E8EEF8)
        `,
        ...extraStyle,
      }}
    >
      {corner && <span className="ph-corner">{corner}</span>}
      <span className="ph-label">{label}</span>
    </div>
  )
}

// ── KpiCard ──────────────────────────────────────────────────────
export function KpiCard({ label, value, sub, trend, trendKind, icon }) {
  const trendStyle = {
    up:     { bg: 'rgba(21,128,61,0.10)', ink: 'var(--ok-ink)' },
    urgent: { bg: 'var(--risk-bg)',        ink: 'var(--risk-ink)' },
    badge:  { bg: 'var(--accent-soft)',    ink: 'var(--primary)' },
  }[trendKind] || { bg: 'var(--bg-2)', ink: 'var(--muted)' }

  return (
    <div className="card" style={{ padding: 22, display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 13, color: 'var(--muted)', fontWeight: 500 }}>{label}</span>
        {trend && (
          <span style={{ fontSize: 11, fontWeight: 700, padding: '3px 8px', borderRadius: 999, background: trendStyle.bg, color: trendStyle.ink }}>
            {trend}
          </span>
        )}
        {icon}
      </div>
      <div style={{ fontSize: 34, fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--ink)', lineHeight: 1.1, marginTop: 4 }}>
        {value}
      </div>
      <div style={{ fontSize: 12, color: 'var(--muted)' }}>{sub}</div>
    </div>
  )
}

// ── SectionHeader ────────────────────────────────────────────────
export function SectionHeader({ icon, label, step }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
      {step && (
        <span className="mono" style={{ fontSize: 11, color: 'var(--muted-2)', fontWeight: 600, letterSpacing: '0.1em' }}>
          {step}
        </span>
      )}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--muted)' }}>
        {icon}
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>{label}</span>
      </div>
      <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
    </div>
  )
}
