import { useState, useEffect } from 'react'
import ParticleSphere from './ParticleSphere'

// Login screen phases: sphere → auth → verifying → success → exploding → (onLoginDone)

function BackIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path d="M15 6l-6 6 6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function SsoIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path d="M12 2l9 4v6c0 5-3.5 9-9 10-5.5-1-9-5-9-10V6l9-4z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M8.5 12l2.5 2.5L15.5 10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="11" fill="#1E3A8A" />
      <path d="M7 12.5l3.2 3.2L17 9" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export default function Login({ onLoginDone }) {
  const [phase, setPhase] = useState('sphere')
  const [hover, setHover] = useState(false)
  const [email, setEmail] = useState('a.mendel@riversidefh.med')
  const [password, setPassword] = useState('••••••••••')
  const [statusText, setStatusText] = useState('Verifying credentials')

  // verifying → success
  useEffect(() => {
    if (phase !== 'verifying') return
    const seq = [
      { t: 0,    text: 'Verifying credentials' },
      { t: 1100, text: 'Establishing secure session' },
      { t: 2300, text: 'Loading patient cohort' },
    ]
    const timers = seq.map(s => setTimeout(() => setStatusText(s.text), s.t))
    const done = setTimeout(() => setPhase('success'), 3400)
    return () => { timers.forEach(clearTimeout); clearTimeout(done) }
  }, [phase])

  // success → leaving (quick opacity fade, no particle explosion)
  useEffect(() => {
    if (phase !== 'success') return
    const r = setTimeout(() => setPhase('leaving'), 900)
    return () => clearTimeout(r)
  }, [phase])

  // leaving → dashboard. The .screen fade is 400ms; hand off just after it.
  useEffect(() => {
    if (phase !== 'leaving') return
    const r = setTimeout(() => onLoginDone(), 420)
    return () => clearTimeout(r)
  }, [phase, onLoginDone])

  const enterFromSphere = () => { setHover(false); setPhase('auth') }
  const submit = (e) => { e.preventDefault(); setPhase('verifying') }
  const backToSphere = () => setPhase('sphere')

  const sphereDim     = phase === 'auth'
  const sphereVisible = true
  const showAuthCard  = phase === 'auth'
  const showStatusOverlay = phase === 'verifying' || phase === 'success' || phase === 'leaving'

  const overlayText =
    phase === 'verifying'                          ? statusText :
    phase === 'success' || phase === 'leaving'     ? 'Welcome back, Dr. Mendel' :
    ''

  return (
    <div className={`screen phase-${phase}`} style={{ background: '#E2EBF6' }}>
      {/* SPHERE LAYER */}
      {sphereVisible && (
        <div className={`stage ${sphereDim ? 'dim' : ''}`}>
          <div className="sphere-wrap">
            <ParticleSphere
              state={phase === 'success' || phase === 'leaving' ? 'success' : 'idle'}
              hovering={hover && phase === 'sphere'}
            />

            {/* Logo — clickable only on sphere phase */}
            <button
              type="button"
              className={`logo-cta ${phase !== 'sphere' ? 'is-disabled' : ''}`}
              onMouseEnter={() => setHover(true)}
              onMouseLeave={() => setHover(false)}
              onClick={() => phase === 'sphere' && enterFromSphere()}
              aria-label="Sign in to Raresight"
              disabled={phase !== 'sphere'}
            >
              <img
                src="/raresight-logo.png"
                alt="raresight"
                draggable="false"
                style={{ width: 245, height: 92, padding: 0, margin: '0 0 -7px', objectFit: 'contain' }}
              />
            </button>

            {/* Status overlay */}
            {showStatusOverlay && (
              <div className={`status-overlay ${phase === 'success' || phase === 'leaving' ? 'is-success' : ''}`} key={overlayText}>
                {(phase === 'success' || phase === 'leaving') && <CheckIcon />}
                <span>{overlayText}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* LOGIN FORM */}
      {showAuthCard && (
        <div className="auth-layer">
          <form className="login-card" onSubmit={submit} style={{ position: 'relative' }}>
            <button type="button" className="back" onClick={backToSphere} aria-label="Back">
              <BackIcon />
            </button>

            <div className="login-logo" style={{ width: 337, height: 64 }}>
              <img
                src="/raresight-logo.png"
                alt="raresight"
                style={{ objectFit: 'cover', height: 52, width: 196, padding: 0, margin: '21.6px 0 0 -17.6px' }}
              />
            </div>

            <h1>Clinician sign in</h1>
            <p className="sub">Secure access to your patient cohort and triage queue.</p>

            <label className="field">
              <span>Work email</span>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                autoComplete="email"
                required
              />
            </label>

            <label className="field">
              <span>Password</span>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                autoComplete="current-password"
                required
              />
            </label>

            <div className="row-between">
              <label className="check">
                <input type="checkbox" defaultChecked /> Trust this device
              </label>
              <a href="#" onClick={e => e.preventDefault()}>Forgot password</a>
            </div>

            <button type="submit" className="primary">Sign in</button>

            <div className="divider"><span>or</span></div>
            <button type="button" className="ghost" onClick={submit}>
              <SsoIcon /> Continue with hospital SSO
            </button>

            <p className="legal">Protected health information. Authorized clinicians only.</p>
          </form>
        </div>
      )}
    </div>
  )
}
