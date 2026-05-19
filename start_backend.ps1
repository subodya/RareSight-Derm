# Run from repo root: .\start_backend.ps1
# Starts the FastAPI backend on http://localhost:8000

$env:PYTHONPATH = $PSScriptRoot
uvicorn src.app.backend.main:app --reload --host 0.0.0.0 --port 8000
