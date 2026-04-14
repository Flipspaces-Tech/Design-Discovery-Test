# Google Sheets Downloader - PowerShell Script
# This script downloads all sheets from a Google Sheets document

param(
    [string]$SheetUrl = "https://docs.google.com/spreadsheets/d/10nrNEmq1E0BUIwYeQLgjyiqsk9e5FwHGXTTN2LMQBHU/edit",
    [string]$OutputDir = "C:\Users\admin\Documents\VIZWALK-AI\new folder\downloaded_sheets",
    [string]$Format = "csv"
)

Write-Host "============================================================"
Write-Host "Google Sheets Downloader" -ForegroundColor Cyan
Write-Host "============================================================"
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation"
    Read-Host "Press Enter to exit"
    exit 1
}

# Install required packages if needed
Write-Host "Checking required packages..." -ForegroundColor Yellow
python -c "import requests, pandas" 2>&1 | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing required packages (requests, pandas)..." -ForegroundColor Yellow
    pip install requests pandas
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install required packages" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host "✅ All packages installed" -ForegroundColor Green
Write-Host ""

# Get the directory of this script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Display parameters
Write-Host "📊 Sheet URL: $SheetUrl"
Write-Host "📁 Output Directory: $OutputDir"
Write-Host "📄 Format: $Format"
Write-Host ""

# Create output directory if it doesn't exist
if (!(Test-Path -Path $OutputDir)) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Host "📁 Created directory: $OutputDir" -ForegroundColor Green
}

# Run the downloader
Write-Host "⬇️  Starting download..." -ForegroundColor Cyan
Write-Host ""

python "$scriptDir\google_sheets_downloader.py" "$SheetUrl" "$OutputDir" "$Format"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Download completed successfully!" -ForegroundColor Green
    Write-Host "📂 Opening folder: $OutputDir" -ForegroundColor Cyan
    Write-Host ""
    
    # Open the output folder in Explorer
    Start-Process "explorer.exe" -ArgumentList $OutputDir
} else {
    Write-Host ""
    Write-Host "❌ An error occurred during download" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit"