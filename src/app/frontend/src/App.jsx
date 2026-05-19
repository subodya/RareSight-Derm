import { useState, useEffect } from 'react'
import Login from './login/Login'
import { Sidebar, TopBar } from './components/common'
import Dashboard from './screens/Dashboard'
import Scan from './screens/Scan'
import Library from './screens/Library'
import Profile from './screens/Profile'
import { Plus } from './icons'

const TITLES = {
  dashboard: 'Dashboard',
  scan: 'Scan & Diagnose',
  library: 'Disease Library',
  profile: 'Profile',
}

export default function App() {
  const [appPhase, setAppPhase]       = useState('login')
  const [route, setRoute]             = useState('dashboard')
  const [systemStatus, setSystemStatus] = useState('checking')
  const [classes, setClasses]         = useState({})

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(() => {
        setSystemStatus('online')
        return fetch('/api/classes')
      })
      .then(r => r.json())
      .then(setClasses)
      .catch(() => setSystemStatus('offline'))
  }, [])

  if (appPhase === 'login') {
    return <Login onLoginDone={() => setAppPhase('app')} />
  }

  const primaryAction = route === 'dashboard' ? (
    <button className="btn btn-primary" onClick={() => setRoute('scan')} style={{ padding: '8px 16px', fontSize: 13 }}>
      <Plus size={14} /> New Scan
    </button>
  ) : null

  return (
    <div className="app">
      <Sidebar route={route} setRoute={setRoute} />
      <div className="main">
        <TopBar title={TITLES[route]} primaryAction={primaryAction} systemStatus={systemStatus} />
        <div key={route} className="fade-up">
          {route === 'dashboard' && <Dashboard setRoute={setRoute} />}
          {route === 'scan'      && <Scan classes={classes} systemStatus={systemStatus} />}
          {route === 'library'   && <Library />}
          {route === 'profile'   && <Profile />}
        </div>
      </div>
    </div>
  )
}
