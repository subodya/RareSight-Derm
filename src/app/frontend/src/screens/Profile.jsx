import { useState } from 'react'
import { User, Stethoscope, Settings, Shield, Award, Logout, Edit,
  Hash, Mail, Phone, MapPin, Calendar, Heart, Check, CheckCircle,
  Upload, Crosshair, Cpu, Users, Sparkle } from '../icons'
import { Avatar } from '../components/common'
import { PROFILE } from '../data'

export default function Profile() {
  const [section, setSection] = useState('account')

  const navItems = [
    { id: 'account', label: 'Account', Icon: User },
    { id: 'practice', label: 'Practice', Icon: Stethoscope },
    { id: 'preferences', label: 'Preferences', Icon: Settings },
    { id: 'security', label: 'Security & Audit', Icon: Shield },
    { id: 'achievements', label: 'Achievements', Icon: Award },
  ]

  return (
    <div className="page fade-up">
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 28 }}>
        {/* LEFT: profile card + nav */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card" style={{ padding: 24, textAlign: 'center' }}>
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 14 }}>
              <Avatar name={PROFILE.name} size={88} />
            </div>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 2 }}>{PROFILE.name}</h2>
            <div style={{ fontSize: 12, color: 'var(--muted)' }}>{PROFILE.title}</div>
            <div style={{ marginTop: 10, padding: '4px 12px', borderRadius: 999, background: 'var(--ok-bg)', color: 'var(--ok-ink)', fontSize: 11, fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <Shield size={12} strokeWidth={2} /> Verified Clinician
            </div>
            <div className="divider" style={{ margin: '18px 0' }} />
            <div style={{ display: 'flex', justifyContent: 'space-around', textAlign: 'center' }}>
              <div>
                <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--ink)' }}>{PROFILE.scans.toLocaleString()}</div>
                <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Scans</div>
              </div>
              <div>
                <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--ink)' }}>{PROFILE.confirmed.toLocaleString()}</div>
                <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Confirmed</div>
              </div>
              <div>
                <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--ink)' }}>{PROFILE.referrals}</div>
                <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Referrals</div>
              </div>
            </div>
          </div>

          <div className="card" style={{ padding: 8 }}>
            {navItems.map(({ id, label, Icon }) => {
              const active = section === id
              return (
                <button
                  key={id}
                  onClick={() => setSection(id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
                    borderRadius: 10, width: '100%', textAlign: 'left',
                    color: active ? 'var(--primary)' : 'var(--ink-2)',
                    background: active ? 'var(--accent-soft)' : 'transparent',
                    fontWeight: active ? 600 : 500, fontSize: 13,
                  }}
                >
                  <Icon size={16} strokeWidth={active ? 1.8 : 1.6} />
                  {label}
                </button>
              )
            })}
            <div className="divider" style={{ margin: '6px 8px' }} />
            <button style={{
              display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
              borderRadius: 10, width: '100%', textAlign: 'left',
              color: 'var(--risk-ink)', fontWeight: 500, fontSize: 13,
            }}>
              <Logout size={16} /> Sign Out
            </button>
          </div>
        </div>

        {/* RIGHT: content */}
        <div>
          {section === 'account'      && <AccountSection />}
          {section === 'practice'     && <PracticeSection />}
          {section === 'preferences'  && <PreferencesSection />}
          {section === 'security'     && <SecuritySection />}
          {section === 'achievements' && <AchievementsSection />}
        </div>
      </div>
    </div>
  )
}

function SectionTitle({ title, sub, action }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 22 }}>
      <div>
        <h2 style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.01em', marginBottom: 4 }}>{title}</h2>
        <p style={{ fontSize: 13, color: 'var(--muted)' }}>{sub}</p>
      </div>
      {action}
    </div>
  )
}

function Field({ label, value, icon, mono }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>{label}</span>
      <div style={{
        padding: '11px 14px', borderRadius: 10, background: 'var(--bg)', border: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', gap: 10,
        color: 'var(--ink)', fontSize: 14,
        fontFamily: mono ? "'JetBrains Mono',monospace" : 'inherit',
      }}>
        {icon}
        <span>{value}</span>
      </div>
    </div>
  )
}

function AccountSection() {
  return (
    <div className="card" style={{ padding: 28 }}>
      <SectionTitle
        title="Account"
        sub="Personal information and credentials on file"
        action={<button className="btn btn-ghost"><Edit size={13} /> Edit</button>}
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        <Field label="Full Name"    value={PROFILE.name}    icon={<User size={14} style={{ color: 'var(--muted)' }} />} />
        <Field label="License ID"   value={PROFILE.license} icon={<Hash size={14} style={{ color: 'var(--muted)' }} />} mono />
        <Field label="Email"        value={PROFILE.email}   icon={<Mail size={14} style={{ color: 'var(--muted)' }} />} />
        <Field label="Phone"        value={PROFILE.phone}   icon={<Phone size={14} style={{ color: 'var(--muted)' }} />} />
        <Field label="Specialty"    value={PROFILE.title}   icon={<Stethoscope size={14} style={{ color: 'var(--muted)' }} />} />
        <Field label="Member Since" value={PROFILE.joined}  icon={<Calendar size={14} style={{ color: 'var(--muted)' }} />} />
      </div>
      <div className="divider" style={{ margin: '26px 0' }} />
      <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 14 }}>Diagnostic Performance</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
        <PerfCard label="Avg. AI Confidence"   value="94.2%" sub="Top-1 prediction" />
        <PerfCard label="Confirmation Rate"     value="95.4%" sub="Diagnoses confirmed without revision" />
        <PerfCard label="Time per Case"         value="3m 41s" sub="Median triage time" />
      </div>
    </div>
  )
}

function PerfCard({ label, value, sub }) {
  return (
    <div style={{ padding: 16, borderRadius: 12, background: 'var(--bg)' }}>
      <div style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--ink)', letterSpacing: '-0.01em', marginTop: 6 }}>{value}</div>
      <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>{sub}</div>
    </div>
  )
}

function PracticeSection() {
  return (
    <div className="card" style={{ padding: 28 }}>
      <SectionTitle title="Practice" sub="Your clinic and patient population" />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 24 }}>
        <Field label="Practice"      value={PROFILE.practice}        icon={<Stethoscope size={14} style={{ color: 'var(--muted)' }} />} />
        <Field label="Location"      value={PROFILE.location}        icon={<MapPin size={14} style={{ color: 'var(--muted)' }} />} />
        <Field label="Patient Panel" value="412 active patients"     icon={<Users size={14} style={{ color: 'var(--muted)' }} />} />
        <Field label="Case Focus"    value="68% dermatology consults" icon={<Heart size={14} style={{ color: 'var(--muted)' }} />} />
      </div>
      <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Specialist Network</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {[
          { name: 'Dr. Priya Shankar', spec: 'Dermatopathology', dist: '14 mi' },
          { name: 'Dr. Owen Tate',     spec: 'Mohs Surgery',     dist: '22 mi' },
          { name: 'Dr. Mei Lin',       spec: 'Melanoma Oncology', dist: '38 mi' },
        ].map((s, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 14, padding: 14, borderRadius: 12, background: 'var(--bg)' }}>
            <Avatar name={s.name} size={40} color="var(--accent)" />
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 14, fontWeight: 600 }}>{s.name}</div>
              <div style={{ fontSize: 12, color: 'var(--muted)' }}>{s.spec} · {s.dist}</div>
            </div>
            <button className="btn btn-soft" style={{ padding: '6px 14px', fontSize: 12 }}>Refer</button>
          </div>
        ))}
      </div>
    </div>
  )
}

function PreferencesSection() {
  const [autoRun, setAutoRun] = useState(true)
  const [referThreshold, setReferThreshold] = useState(85)
  const [language, setLanguage] = useState('English (UK)')

  return (
    <div className="card" style={{ padding: 28 }}>
      <SectionTitle title="Preferences" sub="Configure how RareSight behaves for your workflow" />
      <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
        <Toggle
          label="Auto-run analysis on upload"
          sub="Skip the manual 'Run Analysis' step when a scan is dropped in"
          checked={autoRun}
          onChange={setAutoRun}
        />
        <Toggle
          label="Confidence calibration alerts"
          sub="Warn when entropy is high or top-1 confidence falls below threshold"
          checked={true}
        />
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>Specialist referral threshold</div>
          <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 14 }}>
            Cases above <span className="mono" style={{ color: 'var(--primary)', fontWeight: 700 }}>{referThreshold}%</span> top-1 confidence trigger an auto-referral suggestion
          </div>
          <input
            type="range" min="60" max="99" value={referThreshold}
            onChange={e => setReferThreshold(+e.target.value)}
            style={{ width: '100%', accentColor: 'var(--primary)' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--muted-2)', marginTop: 4 }} className="mono">
            <span>60%</span><span>80%</span><span>99%</span>
          </div>
        </div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>Interface language</div>
          <select value={language} onChange={e => setLanguage(e.target.value)} style={{
            padding: '10px 14px', borderRadius: 10, border: '1px solid var(--border)',
            background: 'var(--bg)', fontSize: 14, color: 'var(--ink)', width: 280,
          }}>
            <option>English (UK)</option>
            <option>English (US)</option>
            <option>Sinhala</option>
            <option>Tamil</option>
            <option>Hindi</option>
          </select>
        </div>
      </div>
    </div>
  )
}

function Toggle({ label, sub, checked: initialChecked = false, onChange }) {
  const [checked, setChecked] = useState(initialChecked)
  const handle = () => { const next = !checked; setChecked(next); onChange?.(next) }
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 14 }}>
      <div>
        <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 2 }}>{label}</div>
        <div style={{ fontSize: 12, color: 'var(--muted)' }}>{sub}</div>
      </div>
      <button onClick={handle} style={{
        width: 44, height: 24, borderRadius: 999,
        background: checked ? 'var(--primary)' : 'var(--border-strong)',
        position: 'relative', transition: 'background .15s', flexShrink: 0,
      }}>
        <span style={{
          position: 'absolute', top: 3, left: checked ? 23 : 3,
          width: 18, height: 18, borderRadius: 999, background: '#fff',
          transition: 'left .2s', boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
        }} />
      </button>
    </div>
  )
}

function SecuritySection() {
  return (
    <div className="card" style={{ padding: 28 }}>
      <SectionTitle title="Security & Audit" sub="Recent activity on your RareSight account" />
      <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Recent Sign-Ins</h3>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {[
          { device: 'MacBook Pro · Chrome',  loc: 'Colombo, LK', time: 'Just now',    current: true },
          { device: 'iPhone 15 · Safari',    loc: 'Colombo, LK', time: '2 hours ago' },
          { device: 'iPad · RareSight App',  loc: 'Colombo, LK', time: 'Yesterday' },
          { device: 'Surface Pro · Edge',    loc: 'Kandy, LK',   time: 'May 12' },
        ].map((s, i) => (
          <div key={i} style={{
            display: 'flex', alignItems: 'center', gap: 14, padding: '14px 0',
            borderBottom: i < 3 ? '1px solid var(--border)' : 'none',
          }}>
            <div style={{ width: 40, height: 40, borderRadius: 10, background: 'var(--bg-2)', display: 'grid', placeItems: 'center', color: 'var(--primary)' }}>
              <Cpu size={18} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600 }}>{s.device}</div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>{s.loc}</div>
            </div>
            <div className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>{s.time}</div>
            {s.current && <span className="pill" style={{ background: 'var(--ok-bg)', color: 'var(--ok-ink)' }}>Current</span>}
          </div>
        ))}
      </div>
      <div className="divider" style={{ margin: '22px 0' }} />
      <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 14 }}>Two-Factor Authentication</h3>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: 16, borderRadius: 12, background: 'var(--ok-bg)' }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <CheckCircle size={20} style={{ color: 'var(--ok-ink)' }} />
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--ok-ink)' }}>2FA enabled via Authenticator app</div>
            <div style={{ fontSize: 11, color: 'var(--ok-ink)', opacity: 0.8 }}>Last verified May 14, 2026</div>
          </div>
        </div>
        <button className="btn btn-ghost" style={{ background: 'rgba(255,255,255,0.6)' }}>Manage</button>
      </div>
    </div>
  )
}

function AchievementsSection() {
  const items = [
    { name: 'First 100 Diagnoses',    sub: 'Confirmed 100+ AI-assisted diagnoses',               earned: true,  Icon: Award,      progress: 100 },
    { name: 'Malignancy Detector',    sub: 'Detected 5+ confirmed malignant lesions',             earned: true,  Icon: Sparkle,    progress: 100 },
    { name: 'Verified Contributor',   sub: 'Submitted anonymized cases to the library',           earned: true,  Icon: Upload,     progress: 100 },
    { name: 'Calibrated Clinician',   sub: 'Avg. confidence within ±5% of confirmation rate',    earned: true,  Icon: Crosshair,  progress: 100 },
    { name: 'Dermoscopy Expert',      sub: 'Analyse 50 dermoscopy cases with >90% confidence',   earned: false, Icon: Heart,      progress: 78 },
    { name: 'Mentor',                 sub: 'Train another clinician on RareSight',                earned: false, Icon: Users,      progress: 0 },
  ]

  return (
    <div className="card" style={{ padding: 28 }}>
      <SectionTitle title="Achievements" sub="Milestones on your RareSight journey" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14 }}>
        {items.map((it, i) => (
          <div key={i} style={{
            padding: 18, borderRadius: 14, opacity: it.earned ? 1 : 0.6,
            background: it.earned ? 'var(--bg)' : 'transparent',
            border: '1px solid var(--border)', display: 'flex', gap: 14,
          }}>
            <div style={{
              width: 48, height: 48, borderRadius: 12, flexShrink: 0,
              background: it.earned ? 'var(--primary)' : 'var(--bg-2)',
              color: it.earned ? '#fff' : 'var(--muted)',
              display: 'grid', placeItems: 'center',
            }}>
              <it.Icon size={22} strokeWidth={1.6} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 14, fontWeight: 700 }}>{it.name}</div>
              <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2, marginBottom: 8 }}>{it.sub}</div>
              {it.earned ? (
                <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--ok-ink)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <Check size={12} strokeWidth={2.2} /> Earned
                </span>
              ) : (
                <div className="bar-track" style={{ height: 6 }}>
                  <div className="bar-fill" style={{ width: `${it.progress}%` }} />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
