import requests
import os
import re
from pathlib import Path
import json

def get_sheet_metadata(spreadsheet_id):
    """
    Get all sheet tab names and their IDs from Google Sheets.
    Uses a clever workaround to extract metadata without official API key.
    """
    try:
        # Try to get sheet metadata via the edit URL which contains tab info
        url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Extract sheet names and gids from the page HTML
            # Look for pattern: gid=12345&title=SheetName
            import re
            
            # Pattern to find sheet information in the HTML
            # Google Sheets includes this data in the page
            patterns = [
                r'"gid":(\d+),"title":"([^"]+)"',
                r'gid=(\d+)[^>]*>([^<]+)<',
            ]
            
            sheets = {}
            for pattern in patterns:
                matches = re.findall(pattern, response.text)
                if matches:
                    for match in matches:
                        gid = match[0]
                        title = match[1]
                        sheets[int(gid)] = title
            
            return sheets
    except:
        pass
    
    return None

def find_all_sheets_by_scanning(spreadsheet_id):
    """
    Fallback method: scan for all sheets by trying gid values.
    Also tries to find sheet names from the download response headers.
    """
    print("Scanning for all sheets...")
    
    valid_sheets = {}
    
    # Scan a wide range of gid values
    for gid in range(2000):
        try:
            csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
            response = requests.get(csv_url, timeout=5)
            
            if response.status_code == 200:
                content = response.text.strip()
                
                # Check if this sheet has data
                if len(content) > 50:
                    lines = [l for l in content.split('\n') if l.strip()]
                    rows = len(lines) - 1
                    
                    if rows > 0:
                        # Try to extract sheet name from response headers
                        sheet_name = f"Sheet_{gid}"
                        
                        # Check if name is in Content-Disposition header
                        if 'Content-Disposition' in response.headers:
                            disp = response.headers['Content-Disposition']
                            match = re.search(r'filename="([^"]+)"', disp)
                            if match:
                                sheet_name = match.group(1).replace('.csv', '')
                        
                        valid_sheets[gid] = {
                            'name': sheet_name,
                            'rows': rows
                        }
                        print(f"   ✓ gid={gid}: '{sheet_name}' ({rows} rows)")
        except:
            pass
        
        # Show progress
        if (gid + 1) % 250 == 0:
            print(f"   (Scanned {gid + 1} gid values...)")
    
    return valid_sheets

def download_all_sheets(sheets_url, output_folder="downloaded_sheets"):
    """
    Main function to download all sheets with their actual names.
    """
    
    print("=" * 70)
    print("AUTO-DOWNLOAD ALL SHEETS FROM GOOGLE SHEETS")
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
    
    # Try to get sheet metadata first
    print("🔍 Detecting sheet tabs...\n")
    sheets_info = get_sheet_metadata(spreadsheet_id)
    
    if not sheets_info:
        print("⚠️  Could not get sheet names from metadata.")
        print("   Using fallback method (scanning gid values)...\n")
        sheets_data = find_all_sheets_by_scanning(spreadsheet_id)
        
        # Convert to standard format
        sheets_info = {}
        for gid, data in sheets_data.items():
            sheets_info[gid] = data['name']
    else:
        print("✓ Found sheets from metadata:\n")
        for gid, name in sheets_info.items():
            print(f"   • gid={gid}: '{name}'")
    
    if not sheets_info:
        print("\n❌ No sheets found")
        return False
    
    print(f"\n✓ Total sheets found: {len(sheets_info)}\n")
    
    # Download each sheet
    print("=" * 70)
    print("DOWNLOADING SHEETS")
    print("=" * 70)
    print()
    
    downloaded = 0
    failed = 0
    sheet_mapping = {}
    
    for idx, (gid, sheet_name) in enumerate(sorted(sheets_info.items()), 1):
        try:
            csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
            
            # Sanitize sheet name for filename
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', sheet_name)
            
            print(f"[{idx}/{len(sheets_info)}] Downloading: '{sheet_name}'...", end=" ", flush=True)
            
            response = requests.get(csv_url, timeout=15)
            
            if response.status_code == 200:
                # Save with meaningful name
                filename = f"{safe_name}.csv"
                filepath = os.path.join(output_folder, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Count rows
                rows = len([line for line in response.text.split('\n') if line.strip()])
                print(f"✓ ({rows} rows)")
                
                downloaded += 1
                sheet_mapping[gid] = {
                    'original_name': sheet_name,
                    'filename': filename,
                    'rows': rows
                }
            else:
                print(f"✗ Failed")
                failed += 1
                
        except Exception as e:
            print(f"✗ Error: {str(e)[:40]}")
            failed += 1
    
    # Save sheet mapping for reference
    mapping_file = os.path.join(output_folder, "sheet_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(sheet_mapping, f, indent=2)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 70}\n")
    
    print(f"✅ Downloaded: {downloaded} sheets")
    if failed > 0:
        print(f"❌ Failed: {failed} sheets")
    
    print(f"\n📁 CSV files saved in: {output_folder}/")
    print(f"\nFiles created:")
    
    for gid, info in sorted(sheet_mapping.items()):
        filepath = os.path.join(output_folder, info['filename'])
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  • {info['filename']} ({info['rows']} rows, {size} bytes)")
            print(f"    └─ Original name: '{info['original_name']}'")
    
    print(f"\n📋 Sheet mapping saved as: sheet_mapping.json")
    print(f"\n🚀 Next step:")
    print(f"   Run: python process_local_sheets.py")
    print(f"   It will process all CSV files from the '{output_folder}' folder\n")
    
    return True

def main():
    print()
    print("AUTO-DOWNLOAD ALL SHEET TABS")
    print("-" * 70)
    print()
    print("This script will:")
    print("  1. Detect ALL sheet tabs in your Google Sheet")
    print("  2. Download each sheet with its proper name")
    print("  3. Save as CSV files")
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
    success = download_all_sheets(sheets_url, output_folder)
    
    if not success:
        print("\n⚠️  Download failed")

if __name__ == "__main__":
    main()