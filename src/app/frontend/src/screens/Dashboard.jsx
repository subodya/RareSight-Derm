import { useState, useEffect, useCallback } from 'react'
import { getDashboard, deleteCase, timeAgo } from '../api'
import { KpiCard, RiskPill } from '../components/common'
import { ArrowRight, Upload, CheckCircle, ArrowUpRight, Folder, Sparkles, Microscope, AlertTriangle, Edit, Trash } from '../icons'
import CaseDetail from './CaseDetail'
import CaseEditModal from './CaseEditModal'

const riskKind = r => (r === 'high' ? 'high' : r === 'medium' ? 'medium' : 'normal')

export default function Dashboard({ setRoute }) {
  const [summary, setSummary] = useState(null)
  const [error, setError]     = useState(null)
  const [openCaseId, setOpenCaseId] = useState(null)
  const [editCaseId, setEditCaseId] = useState(null)

  const load = useCallback(() => {
    getDashboard().then(setSummary).catch(e => setError(e.message))
  }, [])

  useEffect(() => { load() }, [load])

  const handleDelete = async c => {
    if (!window.confirm(`Delete case #${c.case_id}${c.patient_name ? ` (${c.patient_name})` : ''}? This cannot be undone.`)) return
    try {
      await deleteCase(c.id)
      load()
    } catch (e) {
      window.alert(`Could not delete case: ${e.message}`)
    }
  }

  const today = new Date()
  const dateStr = today.toLocaleDateString('en-GB', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })

  if (error) return <OfflineState message={error} />
  if (!summary) return <DashboardSkeleton />

  const { stats, recent_cases, activity, day_counts } = summary

  const kpis = [
    {
      label: "Today's Scans", value: String(stats.today_scans),
      sub: `${stats.total_scans} total scans on record`,
      trend: stats.today_scans > 0 ? 'Active' : 'Quiet', trendKind: 'badge',
    },
    {
      label: 'Pending Review', value: String(stats.pending_review),
      sub: 'Awaiting your confirmation',
      trend: stats.urgent_pending > 0 ? `Urgent: ${stats.urgent_pending}` : 'Not Urgent',
      trendKind: stats.urgent_pending > 0 ? 'urgent' : 'badge',
    },
    {
      label: 'Avg. Confidence',
      value: stats.avg_confidence != null ? `${(stats.avg_confidence * 100).toFixed(1)}%` : '—',
      sub: 'Across all AI diagnostics (temperature-scaled)',
      trend: 'Calibrated', trendKind: 'badge',
    },
  ]

  return (
    <div className="page fade-up">
      {/* Welcome */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontWeight: 700, fontSize: 36, letterSpacing: '-0.02em', color: 'var(--ink)', marginBottom: 6 }}>
          Welcome back, <span style={{ color: 'var(--primary)' }}>Dr. Aris</span>
        </h1>
        <p style={{ color: 'var(--muted)', fontSize: 14 }}>
          {dateStr} · {stats.pending_review > 0
            ? `${stats.pending_review} case${stats.pending_review > 1 ? 's' : ''} need your attention`
            : 'No cases pending review'}
        </p>
      </div>

      {/* KPI strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 36 }}>
        {kpis.map((k, i) => <KpiCard key={i} {...k} />)}
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

      {recent_cases.length === 0 ? (
        <EmptyCases onScan={() => setRoute('scan')} />
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 36 }}>
          {recent_cases.map(c => (
            <CaseCard key={c.id} c={c}
              onOpen={() => setOpenCaseId(c.id)}
              onEdit={() => setEditCaseId(c.id)}
              onDelete={() => handleDelete(c)} />
          ))}
        </div>
      )}

      {/* Lower section */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20 }}>
        <DiagnosticTrend days={day_counts} />
        <ActivityFeed activity={activity} />
      </div>

      {openCaseId != null && (
        <CaseDetail caseId={openCaseId} onClose={() => setOpenCaseId(null)} />
      )}

      {editCaseId != null && (
        <CaseEditModal
          caseId={editCaseId}
          onClose={() => setEditCaseId(null)}
          onSaved={load}
        />
      )}
    </div>
  )
}

function IconButton({ Icon, label, onClick, danger }) {
  return (
    <button
      title={label}
      aria-label={label}
      onClick={onClick}
      style={{
        width: 32, height: 32, borderRadius: 9, display: 'grid', placeItems: 'center',
        background: 'var(--bg-2)', color: 'var(--muted)', transition: 'background .15s, color .15s',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.background = danger ? 'var(--risk-bg)' : 'var(--accent-soft)'
        e.currentTarget.style.color = danger ? 'var(--risk-ink)' : 'var(--primary)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.background = 'var(--bg-2)'
        e.currentTarget.style.color = 'var(--muted)'
      }}
    >
      <Icon size={14} />
    </button>
  )
}

function CaseCard({ c, onOpen, onEdit, onDelete }) {
  return (
    <div
      className="card"
      style={{ display: 'grid', gridTemplateColumns: '180px 1fr', overflow: 'hidden', transition: 'transform .15s, box-shadow .15s' }}
      onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = 'var(--shadow-md)' }}
      onMouseLeave={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'var(--shadow-sm)' }}
    >
      <img
        src={c.thumbnail_url}
        alt={c.top_class_name}
        style={{ width: '100%', height: '100%', minHeight: 184, objectFit: 'cover', display: 'block' }}
      />
      <div style={{ padding: 18, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
          <div style={{ minWidth: 0 }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--ink)', marginBottom: 2 }}>
              {c.patient_name || 'Unnamed Patient'}
            </h3>
            <div className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>
              #{c.case_id}{c.age ? ` · ${c.age}${(c.sex || '').toLowerCase()}` : ''}
            </div>
          </div>
          <RiskPill kind={riskKind(c.risk)} />
        </div>
        <p style={{
          fontSize: 13, color: 'var(--muted)', marginTop: 10, lineHeight: 1.5,
          display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
        }}>
          <span style={{ fontWeight: 600, color: 'var(--ink-2)' }}>{c.top_class_name}</span>
          {' · '}{(c.confidence * 100).toFixed(1)}% confidence
          {c.refer_to_specialist ? ' · referral suggested' : ''}
          {c.localization ? ` · ${c.localization}` : ''}
        </p>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 14, gap: 8 }}>
          <span className="mono" style={{ fontSize: 11, color: 'var(--muted-2)' }}>{timeAgo(c.created_at)}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <IconButton Icon={Edit} label="Edit case" onClick={onEdit} />
            <IconButton Icon={Trash} label="Delete case" onClick={onDelete} danger />
            <button className="btn btn-primary" onClick={onOpen} style={{ padding: '8px 16px', fontSize: 12 }}>
              Review Case <ArrowRight size={13} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function DiagnosticTrend({ days }) {
  const max = Math.max(1, ...days.map(d => d.scans))
  return (
    <div className="card" style={{ padding: 22 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700 }}>Diagnostic Activity</h3>
        <div style={{ display: 'flex', gap: 14, fontSize: 12, color: 'var(--muted)' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: 'var(--primary)' }} /> Scans
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: 'var(--accent-soft)' }} /> Referred
          </span>
        </div>
      </div>
      <p style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 22 }}>Last 14 days</p>

      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, height: 160, paddingBottom: 8, borderBottom: '1px solid var(--border)' }}>
        {days.map((d, i) => (
          <div key={i} title={`${d.date}: ${d.scans} scans, ${d.flags} referred`}
            style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ width: '100%', display: 'flex', flexDirection: 'column-reverse', height: '100%', gap: 2 }}>
              <div style={{ background: 'var(--primary)', height: `${(d.scans / max) * 80}%`, borderRadius: '4px 4px 0 0', minHeight: d.scans > 0 ? 4 : 0 }} />
              <div style={{ background: 'var(--accent-soft)', height: `${(d.flags / max) * 60}%`, borderRadius: '4px 4px 0 0', minHeight: d.flags > 0 ? 4 : 0 }} />
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

function ActivityFeed({ activity }) {
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
      <p style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 14 }}>Recent events</p>
      {activity.length === 0 && (
        <p style={{ fontSize: 13, color: 'var(--muted)' }}>No activity yet — run a scan to get started.</p>
      )}
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {activity.map((a, i) => {
          const Ico = ICONS[a.kind] || ICONS.upload
          return (
            <div key={i} style={{
              display: 'flex', gap: 12, padding: '10px 0',
              borderBottom: i < activity.length - 1 ? '1px solid var(--border)' : 'none',
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
              <div className="mono" style={{ fontSize: 11, color: 'var(--muted-2)', flexShrink: 0 }}>{timeAgo(a.time)}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function EmptyCases({ onScan }) {
  return (
    <div style={{
      border: '1px dashed var(--border-strong)', borderRadius: 16, padding: '56px 20px',
      textAlign: 'center', background: 'var(--surface)', marginBottom: 36,
    }}>
      <div style={{ width: 56, height: 56, borderRadius: 14, background: 'var(--bg-2)', display: 'grid', placeItems: 'center', margin: '0 auto 14px', color: 'var(--primary)' }}>
        <Microscope size={26} strokeWidth={1.5} />
      </div>
      <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--ink)', marginBottom: 4 }}>No scans yet</div>
      <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 16 }}>Run your first scan to start building case history</div>
      <button className="btn btn-primary" onClick={onScan} style={{ padding: '10px 18px', fontSize: 13 }}>
        Run First Scan <ArrowRight size={14} />
      </button>
    </div>
  )
}

function OfflineState({ message }) {
  return (
    <div className="page fade-up">
      <div className="card" style={{ padding: 40, textAlign: 'center' }}>
        <AlertTriangle size={28} style={{ color: 'var(--warn-ink)', margin: '0 auto 12px' }} />
        <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 4 }}>Dashboard unavailable</div>
        <div style={{ fontSize: 13, color: 'var(--muted)' }}>
          Could not reach the RareSight backend ({message}). Start it and reload.
        </div>
      </div>
    </div>
  )
}

function DashboardSkeleton() {
  const block = (h, extra = {}) => (
    <div className="shimmer" style={{ height: h, borderRadius: 14, ...extra }} />
  )
  return (
    <div className="page fade-up">
      <div style={{ marginBottom: 28 }}>{block(44, { width: 420, marginBottom: 8 })}{block(16, { width: 280 })}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 36 }}>
        {block(120)}{block(120)}{block(120)}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 36 }}>
        {block(184)}{block(184)}{block(184)}{block(184)}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20 }}>
        {block(260)}{block(260)}
      </div>
    </div>
  )
}
