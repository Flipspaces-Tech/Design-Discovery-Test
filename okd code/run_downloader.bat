@echo off
REM Google Sheets Downloader - Windows Batch Script
REM This script downloads all sheets from a Google Sheets document

setlocal enabledelayexpansion

echo.
echo ============================================================
echo Google Sheets Downloader
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check if required packages are installed
python -c "import requests, pandas" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install requests pandas
    if errorlevel 1 (
        echo ERROR: Failed to install required packages
        pause
        exit /b 1
    )
)

REM Get the directory of this script
set "SCRIPT_DIR=%~dp0"

REM Default values
set "OUTPUT_DIR=%SCRIPT_DIR%downloaded_sheets"
set "FORMAT=csv"

REM Check if Sheet URL is provided
if "%~1"=="" (
    echo Usage: run_downloader.bat "SHEET_URL" [OUTPUT_DIR] [FORMAT]
    echo.
    echo Example:
    echo run_downloader.bat "https://docs.google.com/spreadsheets/d/ABC123/edit"
    echo run_downloader.bat "https://docs.google.com/spreadsheets/d/ABC123/edit" "C:\my_sheets" xlsx
    echo.
    set /p SHEET_URL="Enter Google Sheets URL: "
) else (
    set "SHEET_URL=%~1"
)

if not "!SHEET_URL!"=="" (
    if not "%~2"=="" set "OUTPUT_DIR=%~2"
    if not "%~3"=="" set "FORMAT=%~3"
    
    echo Sheet URL: !SHEET_URL!
    echo Output Directory: !OUTPUT_DIR!
    echo Format: !FORMAT!
    echo.
    
    python "%SCRIPT_DIR%google_sheets_downloader.py" "!SHEET_URL!" "!OUTPUT_DIR!" "!FORMAT!"
    
    if errorlevel 0 (
        echo.
        echo Press any key to open the output folder...
        pause
        start "" "!OUTPUT_DIR!"
    )
) else (
    echo ERROR: No URL provided
    pause
    exit /b 1
)