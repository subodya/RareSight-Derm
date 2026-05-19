import { useEffect, useRef } from 'react'

export default function ParticleSphere({
  density = 1500,
  speed = 1.0,
  particleSize = 1.6,
  repulsion = 1.0,
  repulseRadius = 70,
  glow = 0.6,
  color = '#1E3A8A',
  accent = '#2563EB',
  radius = 220,
  state = 'idle',
  hovering = false,
  onExplodeDone,
}) {
  const canvasRef = useRef(null)

  // These refs are updated every render so the RAF loop always reads current values
  // without needing to restart the loop when props change.
  const onExplodeDoneRef = useRef(onExplodeDone)
  const stateRef = useRef(state)
  const hoveringRef = useRef(hovering)
  onExplodeDoneRef.current = onExplodeDone
  stateRef.current = state
  hoveringRef.current = hovering

  const animRef = useRef({
    particles: [],
    mouse: { x: -9999, y: -9999, active: false, downAt: 0 },
    rot: { x: 0, y: 0, tx: 0, ty: 0 },
    burst: 0,
    hoverAmt: 0,
    successAmt: 0,
    explode: 0,
    explodeFade: 1,
    explodeFired: false,
    explodeDoneFired: false,
  })

  // Build particles when density changes
  useEffect(() => {
    const N = Math.max(100, Math.floor(density))
    const golden = Math.PI * (Math.sqrt(5) - 1)
    const pts = []
    for (let i = 0; i < N; i++) {
      const y = 1 - (i / (N - 1)) * 2
      const phi = Math.acos(y)
      const theta = (golden * i) % (Math.PI * 2)
      const rJit = 0.94 + Math.random() * 0.10
      const equatorFactor = Math.sin(phi)
      const variance = 0.6 + Math.random() * 0.8
      const driftDir = Math.random() < 0.92 ? 1 : -1
      pts.push({
        theta, phi, rJit,
        drift: 0.00018 * equatorFactor * variance * driftDir,
        ox: 0, oy: 0, oz: 0,
        vx: 0, vy: 0, vz: 0,
        size: 0.55 + Math.random() * 0.9,
        bright: 0.55 + Math.random() * 0.45,
      })
    }
    animRef.current.particles = pts
  }, [density])

  // Pointer handling — stable, no dependencies
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const s = animRef.current

    const onMove = (e) => {
      const rect = canvas.getBoundingClientRect()
      const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left
      const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top
      s.mouse.x = cx
      s.mouse.y = cy
      s.mouse.active = true
      const nx = (cx - rect.width  / 2) / rect.width
      const ny = (cy - rect.height / 2) / rect.height
      s.rot.ty = nx * 0.9
      s.rot.tx = -ny * 0.7
    }
    const onLeave = () => {
      s.mouse.active = false
      s.mouse.x = -9999
      s.mouse.y = -9999
      s.rot.tx = 0
      s.rot.ty = 0
    }
    const onDown = () => { s.mouse.downAt = performance.now() }
    const onUp = () => {
      if (performance.now() - s.mouse.downAt < 300) s.burst = 1.0
    }

    canvas.addEventListener('mousemove', onMove)
    canvas.addEventListener('mouseleave', onLeave)
    canvas.addEventListener('mousedown', onDown)
    canvas.addEventListener('mouseup', onUp)
    canvas.addEventListener('touchmove', onMove, { passive: true })
    canvas.addEventListener('touchstart', (e) => { onDown(); onMove(e) }, { passive: true })
    canvas.addEventListener('touchend', onUp)
    return () => {
      canvas.removeEventListener('mousemove', onMove)
      canvas.removeEventListener('mouseleave', onLeave)
      canvas.removeEventListener('mousedown', onDown)
      canvas.removeEventListener('mouseup', onUp)
      canvas.removeEventListener('touchmove', onMove)
      canvas.removeEventListener('touchend', onUp)
    }
  }, [])

  // Single long-running RAF loop — reads state via refs so it never restarts
  // due to state/hovering prop changes. This is critical: restarting mid-explosion
  // would reset explodeFired and break the callback chain.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const s = animRef.current
    let raf = 0
    const dpr = Math.min(window.devicePixelRatio || 1, 2)

    const fit = () => {
      const rect = canvas.getBoundingClientRect()
      canvas.width  = Math.floor(rect.width  * dpr)
      canvas.height = Math.floor(rect.height * dpr)
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(canvas)

    const colorRGB  = hexToRgb(color)
    const accentRGB = hexToRgb(accent)
    let last = performance.now()

    const tick = (now) => {
      const dt = Math.min(40, now - last)
      last = now

      // Read current state from refs — no closure capture
      const curState    = stateRef.current
      const curHovering = hoveringRef.current

      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)

      const cx = w / 2
      const cy = h / 2
      const R = radius * dpr
      const focal = R * 2.4

      s.hoverAmt   += ((curHovering ? 1 : 0) - s.hoverAmt)   * 0.08
      s.successAmt += ((curState === 'success' ? 1 : 0) - s.successAmt) * 0.06

      if (curState === 'exploding') {
        if (!s.explodeFired) {
          const pts0 = s.particles
          for (let i = 0; i < pts0.length; i++) {
            const p = pts0[i]
            const sinPhi = Math.sin(p.phi)
            const nx = sinPhi * Math.cos(p.theta)
            const ny = Math.cos(p.phi)
            const nz = sinPhi * Math.sin(p.theta)
            const kick = 1.6 + Math.random() * 1.4
            p.vx += nx * kick + (Math.random() - 0.5) * 0.4
            p.vy += ny * kick + (Math.random() - 0.5) * 0.4
            p.vz += nz * kick + (Math.random() - 0.5) * 0.4
          }
          s.explodeFired = true
          s.explode = 0
          s.explodeFade = 1
          s.explodeDoneFired = false
        }
        s.explode = Math.min(1, s.explode + dt / 200)
        if (s.explode > 0.35) {
          s.explodeFade = Math.max(0, s.explodeFade - dt / 900)
        }
        if (s.explodeFade <= 0.01 && !s.explodeDoneFired && onExplodeDoneRef.current) {
          s.explodeDoneFired = true
          const cb = onExplodeDoneRef.current
          setTimeout(cb, 0)  // defer so RAF cleanup doesn't race with React state update
        }
      } else {
        // Reset explosion state if we leave exploding (shouldn't happen in normal flow)
        if (s.explodeFired) {
          s.explodeFired = false
          s.explodeDoneFired = false
          s.explode = 0
          s.explodeFade = 1
          const pts0 = s.particles
          for (let i = 0; i < pts0.length; i++) {
            const p = pts0[i]
            p.ox = p.oy = p.oz = 0
            p.vx = p.vy = p.vz = 0
          }
        }
      }

      s.rot.x += (s.rot.tx - s.rot.x) * 0.05
      s.rot.y += (s.rot.ty - s.rot.y) * 0.05

      const exploding = curState === 'exploding'
      const cosX = Math.cos(s.rot.x), sinX = Math.sin(s.rot.x)
      const cosY = Math.cos(s.rot.y), sinY = Math.sin(s.rot.y)

      s.burst *= 0.93

      const mouseAct = s.mouse.active
      const mx  = s.mouse.x * dpr
      const my  = s.mouse.y * dpr
      const rR  = repulseRadius * dpr
      const rR2 = rR * rR

      const pts  = s.particles
      const proj = new Array(pts.length)
      const expand = 1 + s.hoverAmt * 0.04 + s.successAmt * 0.20

      for (let i = 0; i < pts.length; i++) {
        const p = pts[i]
        p.theta += p.drift * speed * dt
        if (p.theta > Math.PI * 2) p.theta -= Math.PI * 2
        else if (p.theta < 0)      p.theta += Math.PI * 2

        p.ox += p.vx * dt
        p.oy += p.vy * dt
        p.oz += p.vz * dt
        if (exploding) {
          p.vx *= 0.995; p.vy *= 0.995; p.vz *= 0.995
        } else {
          p.vx *= 0.84; p.vy *= 0.84; p.vz *= 0.84
          p.ox *= 0.90; p.oy *= 0.90; p.oz *= 0.90
        }

        const sinPhi = Math.sin(p.phi)
        const bx = sinPhi * Math.cos(p.theta)
        const by = Math.cos(p.phi)
        const bz = sinPhi * Math.sin(p.theta)

        const offScale = exploding ? R : 1
        const rr = R * p.rJit * expand
        let x = bx * rr + p.ox * offScale
        let y = by * rr + p.oy * offScale
        let z = bz * rr + p.oz * offScale

        const x1 = x * cosY + z * sinY
        const z1 = -x * sinY + z * cosY
        const y2 = y * cosX - z1 * sinX
        const z2 = y * sinX + z1 * cosX

        const persp = focal / (focal + z2)
        const sx = cx + x1 * persp
        const sy = cy + y2 * persp

        proj[i] = { sx, sy, z: z2, persp, p }
      }

      if (mouseAct && repulsion > 0.01) {
        for (let i = 0; i < proj.length; i++) {
          const pr = proj[i]
          const dx = pr.sx - mx
          const dy = pr.sy - my
          const d2 = dx * dx + dy * dy
          if (d2 < rR2 && d2 > 0.5) {
            const d = Math.sqrt(d2)
            const t = 1 - d / rR
            const force = t * t * repulsion * 0.55
            const inv = 1 / pr.persp
            pr.p.vx += (dx / d) * force * inv * 1.8
            pr.p.vy += (dy / d) * force * inv * 1.8
          }
        }
      }

      if (s.burst > 0.02) {
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i]
          const sinPhi = Math.sin(p.phi)
          p.vx += sinPhi * Math.cos(p.theta) * s.burst * 14
          p.vy += Math.cos(p.phi) * s.burst * 14
          p.vz += sinPhi * Math.sin(p.theta) * s.burst * 14
        }
      }

      proj.sort((a, b) => b.z - a.z)

      const baseSize = particleSize * dpr
      const fade = s.explodeFade
      for (let i = 0; i < proj.length; i++) {
        const pr = proj[i]
        const p  = pr.p
        const depth = (pr.z + R) / (2 * R)
        const alpha = (0.16 + 0.85 * depth) * p.bright * fade
        const size  = baseSize * p.size * (0.55 + 0.7 * depth) * pr.persp

        if (glow > 0.01 && depth > 0.6) {
          const g = (depth - 0.6) * glow * 0.7
          ctx.fillStyle = `rgba(${accentRGB.r},${accentRGB.g},${accentRGB.b},${g * 0.22})`
          ctx.beginPath()
          ctx.arc(pr.sx, pr.sy, size * 3.2, 0, Math.PI * 2)
          ctx.fill()
        }

        ctx.fillStyle = `rgba(${colorRGB.r},${colorRGB.g},${colorRGB.b},${alpha})`
        ctx.beginPath()
        ctx.arc(pr.sx, pr.sy, size, 0, Math.PI * 2)
        ctx.fill()
      }

      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  // Only restart the loop if visual constants change — NOT state/hovering (those use refs)
  }, [density, speed, particleSize, repulsion, repulseRadius, glow, color, accent, radius])

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
    />
  )
}

function hexToRgb(hex) {
  const m = hex.replace('#', '')
  const n = m.length === 3 ? m.split('').map(c => c + c).join('') : m
  const v = parseInt(n, 16)
  return { r: (v >> 16) & 255, g: (v >> 8) & 255, b: v & 255 }
}
