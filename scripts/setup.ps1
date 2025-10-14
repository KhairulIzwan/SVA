# Setup script for SVA project
Write-Host "🚀 Setting up SVA (Smart Video Assistant) project..." -ForegroundColor Green

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found! Please install Python 3.11+ first." -ForegroundColor Red
    exit 1
}

# Virtual environment setup
Write-Host "Checking for existing virtual environment..." -ForegroundColor Yellow
Set-Location backend

if (Test-Path "venv") {
    Write-Host "✅ Virtual environment already exists, skipping creation" -ForegroundColor Green
} else {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✅ Virtual environment created successfully" -ForegroundColor Green
}

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Change to project directory for dependency installation
Set-Location "c:\Users\kkamsani\OneDrive - Intel Corporation\Desktop\SVA"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "❌ requirements.txt not found!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Dependencies installation complete!" -ForegroundColor Green

# Check if Node.js is installed for frontend
Write-Host "Checking Node.js for frontend..." -ForegroundColor Yellow
node --version
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Node.js found! You can set up frontend later." -ForegroundColor Green
} else {
    Write-Host "⚠️  Node.js not found. Install it later for frontend development." -ForegroundColor Yellow
}

Write-Host "🎉 Project setup complete! Next steps:" -ForegroundColor Green
Write-Host "1. Test with: cd backend && python test_setup.py" -ForegroundColor Cyan
Write-Host "2. Add a test video to data/videos/ folder" -ForegroundColor Cyan
Write-Host "3. Run your first transcription test" -ForegroundColor Cyan