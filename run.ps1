param(
    [switch]$Setup
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($Setup) {
    if (-not (Test-Path ".venv")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Cyan
        python -m venv .venv
    }
    & .\.venv\Scripts\Activate.ps1
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    pip install -r requirements.txt
    Write-Host "Downloading spaCy English model..." -ForegroundColor Cyan
    python -m spacy download en_core_web_sm
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env. Add your free Groq key at https://console.groq.com (optional)." -ForegroundColor Yellow
    }
    Write-Host "Setup complete. Run: .\run.ps1" -ForegroundColor Green
    exit 0
}

if (-not (Test-Path ".venv")) {
    Write-Host "First-time setup needed. Running: .\run.ps1 -Setup" -ForegroundColor Yellow
    & $MyInvocation.MyCommand.Path -Setup
}

& .\.venv\Scripts\Activate.ps1
Write-Host "Starting server at http://localhost:8000" -ForegroundColor Green
python -m uvicorn backend.main:app --reload --port 8000
