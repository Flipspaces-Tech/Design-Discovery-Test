import requests
import os
import re
from pathlib import Path
import time

def find_all_sheet_gids(spreadsheet_id, max_scan=1000):
    """
    Find all sheet gid values by scanning a wider range.
    Google Sheets assigns gid values that aren't always sequential.
    This scans from 0 to max_scan to find all active sheets.
    """
    print("🔍 Scanning for all sheets (this may take a moment)...")
    
    valid_gids = []
    
    for gid in range(max_scan):
        try:
            csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
            response = requests.get(csv_url, timeout=5)
            
            if response.status_code == 200:
                content = response.text.strip()
                
                # Check if this is a valid sheet with data
                if len(content) > 50:  # Has actual content
                    lines = [l for l in content.split('\n') if l.strip()]
                    rows = len(lines) - 1  # Subtract header
                    
                    if rows > 0:  # Has at least 1 row of data
                        valid_gids.append({
                            'gid': gid,
                            'rows': rows
                        })
                        print(f"   ✓ Found sheet: gid={gid} ({rows} rows)")
        except:
            pass
        
        # Show progress every 100 scans
        if (gid + 1) % 100 == 0:
            print(f"   (Scanned {gid + 1} gid values...)")
    
    return valid_gids

def download_all_sheets_as_csv(sheets_url, output_folder="downloaded_sheets"):
    """
    Download all sheets from a Google Sheet as CSV files.
    
    Args:
        sheets_url: Google Sheets URL
        output_folder: Folder to save CSV files
    """
    
    print("=" * 70)
    print("GOOGLE SHEETS DOWNLOADER - FIXED VERSION")
    print("=" * 70)
    print()
    
    # Extract spreadsheet ID
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheets_url)
    if not match:
        print("❌ Invalid Google Sheets URL")
        return False
    
    spreadsheet_id = match.group(1)
    print(f"✓ Spreadsheet ID: {spreadsheet_id}\n")
    
    # Create output folder
    Path(output_folder).mkdir(exist_ok=True)
    print(f"📁 Output folder: {output_folder}/\n")
    
    # Find all sheets
    sheets_found = find_all_sheet_gids(spreadsheet_id, max_scan=1000)
    
    if not sheets_found:
        print("\n❌ No sheets found. Check that:")
        print("   • Google Sheets URL is correct")
        print("   • Sheet is publicly shared")
        return False
    
    print(f"\n✓ Found {len(sheets_found)} sheet(s)\n")
    
    # Download each sheet
    print("=" * 70)
    print("DOWNLOADING SHEETS")
    print("=" * 70)
    print()
    
    downloaded = 0
    failed = 0
    
    for idx, sheet_info in enumerate(sheets_found, 1):
        gid = sheet_info['gid']
        
        try:
            csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
            print(f"[{idx}/{len(sheets_found)}] Downloading sheet (gid={gid})...", end=" ", flush=True)
            
            response = requests.get(csv_url, timeout=15)
            
            if response.status_code == 200:
                # Save as CSV with gid in filename for clarity
                filename = f"sheet_gid_{gid}.csv"
                filepath = os.path.join(output_folder, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Count rows
                rows = len([line for line in response.text.split('\n') if line.strip()])
                print(f"✓ Saved as '{filename}' ({rows} rows)")
                downloaded += 1
                
            else:
                print(f"✗ Failed (Status {response.status_code})")
                failed += 1
                
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            failed += 1
    
    # Summary
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 70}\n")
    
    print(f"✅ Downloaded: {downloaded} sheets")
    print(f"❌ Failed: {failed} sheets")
    print(f"\n📁 CSV files saved in: {output_folder}/")
    print(f"\nFiles created:")
    
    for sheet_info in sheets_found:
        gid = sheet_info['gid']
        filename = f"sheet_gid_{gid}.csv"
        filepath = os.path.join(output_folder, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  • {filename} ({size} bytes)")
    
    print(f"\n🚀 Next step:")
    print(f"   Run: python process_local_sheets.py")
    print(f"   It will process all CSV files from the '{output_folder}' folder\n")
    
    return True

def main():
    print()
    print("Enter your Google Sheets URL:")
    print("Example: https://docs.google.com/spreadsheets/d/YOUR_ID/edit")
    print()
    
    sheets_url = input("📋 Sheets URL: ").strip()
    
    if not sheets_url:
        print("\n❌ No URL provided")
        return
    
    # Optional: specify output folder
    print()
    output_folder = input("📁 Output folder name [default: downloaded_sheets]: ").strip()
    if not output_folder:
        output_folder = "downloaded_sheets"
    
    print()
    
    # Download sheets
    success = download_all_sheets_as_csv(sheets_url, output_folder)
    
    if not success:
        print("\n⚠️  Download failed")

if __name__ == "__main__":
    main()