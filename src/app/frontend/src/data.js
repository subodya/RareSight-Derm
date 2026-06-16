// Static UI constants. All clinical/case/library data now comes from the
// backend API (see api.js); only presentation-layer constants live here.

export const ANALYSIS_STEPS = [
  'Preprocessing image...',
  'Encoding visual features (BiomedCLIP)...',
  'Computing multimodal embeddings...',
  'Matching against prototype library...',
  'Generating attention rollout map...',
  'Calibrating confidence (temperature scaling)...',
]

// Sidebar identity card fallback (Profile screen fetches the live version
// with usage stats from /api/profile).
export const PROFILE = {
  name: 'Dr. Aris Mendel',
  title: 'General Practitioner · Rural Dermatology',
}
