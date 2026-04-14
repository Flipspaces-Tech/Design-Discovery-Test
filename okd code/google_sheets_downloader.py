"""
Google Sheets Downloader
Downloads all sheets from a Google Sheets document as local CSV or Excel files
"""

import os
import re
import requests
import pandas as pd
from urllib.parse import urlparse, parse_qs
import sys


def extract_sheet_id(url):
    """
    Extract the sheet ID from a Google Sheets URL
    
    Supports formats:
    - https://docs.google.com/spreadsheets/d/SHEET_ID/edit#gid=0
    - https://docs.google.com/spreadsheets/d/SHEET_ID/
    """
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    if match:
        return match.group(1)
    return None


def get_sheet_names(sheet_id):
    """
    Fetch sheet names from a Google Sheet by parsing the HTML page
    """
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Extract sheet names and IDs from the HTML
        sheet_pattern = r'"sheetId":(\d+),"title":"([^"]+)"'
        matches = re.findall(sheet_pattern, response.text)
        
        if not matches:
            print("❌ Could not extract sheets. Make sure the sheet is publicly shared.")
            return []
        
        sheets = [(int(gid), name) for gid, name in matches]
        return sheets
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching sheet: {e}")
        return []


def download_sheet_as_csv(sheet_id, gid, sheet_name, output_dir):
    """
    Download a specific sheet as CSV
    """
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Clean sheet name for filename
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', sheet_name)
        filepath = os.path.join(output_dir, f"{clean_name}.csv")
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {sheet_name} → {clean_name}.csv")
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {sheet_name}: {e}")
        return None


def download_sheet_as_excel(sheet_id, gid, sheet_name, output_dir):
    """
    Download a specific sheet as Excel file
    """
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx&gid={gid}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', sheet_name)
        filepath = os.path.join(output_dir, f"{clean_name}.xlsx")
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {sheet_name} → {clean_name}.xlsx")
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {sheet_name}: {e}")
        return None


def download_all_sheets(sheet_url, output_dir="./downloaded_sheets", format="csv"):
    """
    Main function to download all sheets from a Google Sheets document
    
    Args:
        sheet_url (str): URL of the Google Sheets document
        output_dir (str): Directory to save downloaded files (default: ./downloaded_sheets)
        format (str): File format - 'csv' or 'xlsx' (default: csv)
    """
    
    print("=" * 60)
    print("Google Sheets Downloader")
    print("=" * 60)
    
    # Extract sheet ID
    sheet_id = extract_sheet_id(sheet_url)
    if not sheet_id:
        print("❌ Invalid Google Sheets URL format")
        print("Expected format: https://docs.google.com/spreadsheets/d/SHEET_ID/edit")
        return False
    
    print(f"📊 Sheet ID: {sheet_id}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Format: {format.upper()}")
    print()
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")
    
    # Get sheet names
    print("🔍 Fetching sheet names...")
    sheets = get_sheet_names(sheet_id)
    
    if not sheets:
        return False
    
    print(f"✅ Found {len(sheets)} sheet(s):")
    for gid, name in sheets:
        print(f"   • {name}")
    print()
    
    # Download each sheet
    print("⬇️  Downloading sheets...")
    print()
    
    downloaded_files = []
    
    for i, (gid, sheet_name) in enumerate(sheets, 1):
        print(f"[{i}/{len(sheets)}] Downloading: {sheet_name}")
        
        if format.lower() == "xlsx":
            filepath = download_sheet_as_excel(sheet_id, gid, sheet_name, output_dir)
        else:  # Default to CSV
            filepath = download_sheet_as_csv(sheet_id, gid, sheet_name, output_dir)
        
        if filepath:
            downloaded_files.append(filepath)
    
    # Summary
    print()
    print("=" * 60)
    if downloaded_files:
        print(f"✅ Successfully downloaded {len(downloaded_files)} file(s)")
        print(f"📂 Location: {os.path.abspath(output_dir)}")
        print("=" * 60)
        return True
    else:
        print("❌ No files were downloaded")
        print("=" * 60)
        return False


def main():
    """
    Command-line interface
    """
    if len(sys.argv) < 2:
        print("Usage: python google_sheets_downloader.py <SHEET_URL> [OUTPUT_DIR] [FORMAT]")
        print()
        print("Arguments:")
        print("  SHEET_URL    - URL of the Google Sheets document (required)")
        print("  OUTPUT_DIR   - Output directory (default: ./downloaded_sheets)")
        print("  FORMAT       - File format: 'csv' or 'xlsx' (default: csv)")
        print()
        print("Examples:")
        print("  python google_sheets_downloader.py 'https://docs.google.com/spreadsheets/d/ABC123/edit'")
        print("  python google_sheets_downloader.py 'https://docs.google.com/spreadsheets/d/ABC123/edit' ./my_sheets")
        print("  python google_sheets_downloader.py 'https://docs.google.com/spreadsheets/d/ABC123/edit' ./my_sheets xlsx")
        sys.exit(1)
    
    sheet_url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./downloaded_sheets"
    format = sys.argv[3] if len(sys.argv) > 3 else "csv"
    
    download_all_sheets(sheet_url, output_dir, format)


if __name__ == "__main__":
    main()