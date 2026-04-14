import pandas as pd
import json
import numpy as np
from PIL import Image
import clip
import torch
import os
import requests
from io import BytesIO
import re
import time

# Load CLIP model
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Using device: {device}\n")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def download_image_from_url(image_url):
    """Download image from any URL (S3, Google Drive, HTTP, etc.) and return PIL Image"""
    try:
        if pd.isna(image_url) or not isinstance(image_url, str):
            return None
        
        image_url = str(image_url).strip()
        if not image_url.startswith('http'):
            return None
        
        # Handle Google Drive URLs
        if "drive.google.com" in image_url:
            if "/d/" in image_url:
                file_id = image_url.split("/d/")[1].split("/")[0]
            elif "id=" in image_url:
                file_id = image_url.split("id=")[1].split("&")[0]
            else:
                return None
            
            image_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        response = requests.get(image_url, timeout=15, allow_redirects=True)
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
            
    except:
        pass
    
    return None

def embed_image(image):
    """Convert image to embedding using CLIP"""
    try:
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        return image_features.cpu().numpy().flatten().tolist()
    except:
        return None

def extract_image_urls_from_field(images_field):
    """Extract image URLs from the 'images' field"""
    try:
        if pd.isna(images_field):
            return []
        
        images_field = str(images_field).strip()
        
        if not images_field:
            return []
        
        urls = []
        
        # Try JSON array parsing first
        if images_field.startswith('['):
            try:
                parsed = json.loads(images_field)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and 'url' in item:
                            urls.append(item['url'])
                        elif isinstance(item, str) and item.startswith('http'):
                            urls.append(item)
            except:
                pass
        
        # If no URLs found from JSON, try comma-separated
        if not urls and ',' in images_field:
            parts = images_field.split(',')
            for part in parts:
                url = part.strip()
                if url.startswith('http'):
                    urls.append(url)
        
        # If still no URLs, check if it's a single URL
        if not urls and images_field.startswith('http'):
            urls.append(images_field)
        
        return urls
    
    except:
        return []

# ============================================================
# MAIN PROCESS
# ============================================================

def main():
    print("=" * 70)
    print("BOQ PRODUCT CATALOG EMBEDDING - ALL SHEETS")
    print("=" * 70)
    print()
    
    # Get Google Sheets URL
    print("Step 1: Enter your Google Sheets URL")
    print("-" * 70)
    print("Example: https://docs.google.com/spreadsheets/d/YOUR_ID/edit")
    print()
    sheets_url = input("📋 Sheets URL: ").strip()
    
    if not sheets_url:
        print("\n❌ Error: No URL provided")
        exit()
    
    # Extract spreadsheet ID
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheets_url)
    if not match:
        print("\n❌ Error: Invalid Google Sheets URL format")
        exit()
    
    spreadsheet_id = match.group(1)
    print(f"\n✓ Spreadsheet ID: {spreadsheet_id}")
    
    # Get sheet tabs to process
    print("\n\nStep 2: Specify which sheet tabs to process")
    print("-" * 70)
    print("Enter the sheet tab names you want to process (comma-separated)")
    print("Example: Metal Ceiling,Acoustic Ceiling,Stretch Ceiling,Decoratives")
    print()
    print("To process all sheets with gid=0,1,2,3... just press Enter")
    print("(The script will auto-detect available sheets)")
    print()
    
    sheet_input = input("📝 Sheet tabs (or press Enter for auto-detect): ").strip()
    
    if sheet_input:
        # User provided sheet names
        sheet_tabs = []
        for idx, name in enumerate(sheet_input.split(',')):
            name = name.strip()
            if name:
                sheet_tabs.append({
                    'gid': idx,
                    'name': name
                })
    else:
        # Auto-detect by scanning gid values
        print("\n🔍 Auto-detecting available sheets...")
        sheet_tabs = []
        for gid in range(100):
            try:
                csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
                response = requests.get(csv_url, timeout=5)
                
                if response.status_code == 200 and len(response.content) > 100:
                    lines = response.text.split('\n')
                    row_count = len([l for l in lines if l.strip()])
                    
                    if row_count > 0:
                        sheet_tabs.append({
                            'gid': gid,
                            'name': f'Sheet_{gid}'
                        })
                        print(f"   Found: Sheet_{gid} ({row_count} rows)")
            except:
                pass
        
        if not sheet_tabs:
            print("⚠️  No sheets detected. Using default (Sheet_0)...")
            sheet_tabs = [{'gid': 0, 'name': 'Sheet_0'}]
    
    if not sheet_tabs:
        print("\n❌ No sheets specified")
        exit()
    
    print(f"\n✓ Will process {len(sheet_tabs)} sheet(s):")
    for sheet in sheet_tabs:
        print(f"   • {sheet['name']} (gid={sheet['gid']})")
    
    # Column configuration
    print(f"\n✓ Using BOQ column structure:")
    print(f"   • Images column: 'images'")
    print(f"   • Product name: 'item_name'")
    print(f"   • SKU: 'sku_id'")
    print(f"   • Category: 'category'")
    
    print(f"\n{'=' * 70}")
    print("PROCESSING SHEETS")
    print(f"{'=' * 70}\n")
    
    # Create embeddings dictionary
    all_embeddings = {}
    metadata = {}
    category_stats = {}
    
    # Process each sheet
    for sheet_idx, sheet_info in enumerate(sheet_tabs, 1):
        gid = sheet_info['gid']
        sheet_name = sheet_info['name']
        
        print(f"\n[{sheet_idx}/{len(sheet_tabs)}] Processing: {sheet_name} (gid={gid})")
        print("-" * 70)
        
        try:
            csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
            
            print(f"  Fetching data...", end=" ", flush=True)
            df = pd.read_csv(csv_url)
            print(f"✓ {len(df)} rows found")
            
            cols = list(df.columns)
            print(f"  Columns: {', '.join(cols[:5])}... ({len(cols)} total)")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:60]}")
            continue
        
        # Check for required images column
        if 'images' not in df.columns:
            print(f"  ⚠️  'images' column not found")
            print(f"     Skipping this sheet...")
            continue
        
        successful = 0
        failed = 0
        skipped = 0
        
        # Process each product row
        for idx, row in df.iterrows():
            if pd.isna(row.get('item_name')) or pd.isna(row.get('sku_id')):
                skipped += 1
                continue
            
            print(f"  [{idx+1}/{len(df)}] ", end="", flush=True)
            
            image_urls = extract_image_urls_from_field(row['images'])
            
            if not image_urls:
                print(f"✗")
                failed += 1
                continue
            
            product_name = str(row.get('item_name', 'Unknown')).strip()
            sku = str(row.get('sku_id', f'PROD_{idx}')).strip()
            category = str(row.get('category', sheet_name)).strip()
            color = str(row.get('color', 'Unknown')).strip()
            style = str(row.get('item_style', 'Unknown')).strip()
            brand = str(row.get('brand', 'Unknown')).strip()
            
            primary_image_url = image_urls[0]
            unique_key = f"{category}_{sku}_{product_name[:15]}"
            
            print(f"{sku[:10]}...", end=" ", flush=True)
            
            image = download_image_from_url(primary_image_url)
            if image is None:
                print("✗")
                failed += 1
                continue
            
            print("✓", end=" ", flush=True)
            
            embedding = embed_image(image)
            if embedding is None:
                print("✗")
                failed += 1
                continue
            
            all_embeddings[unique_key] = {
                "embedding": embedding,
                "category": category
            }
            
            metadata[unique_key] = {
                "sku": sku,
                "product_name": product_name,
                "category": category,
                "color": color,
                "style": style,
                "brand": brand,
                "image_url": primary_image_url,
                "all_image_urls": image_urls,
                "row_index": idx
            }
            
            print("✓")
            successful += 1
        
        category_stats[sheet_name] = {
            "gid": gid,
            "total_rows": len(df),
            "embedded": successful,
            "failed": failed,
            "skipped": skipped
        }
        
        print(f"\n  ✓ Summary: {successful} embedded, {failed} failed, {skipped} skipped")
    
    # Save results
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print(f"{'=' * 70}\n")
    
    with open("all_embeddings.json", "w") as f:
        json.dump(all_embeddings, f)
    
    with open("products_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    with open("embedding_stats.json", "w") as f:
        json.dump(category_stats, f, indent=2)
    
    total_embedded = sum(stats["embedded"] for stats in category_stats.values())
    total_failed = sum(stats["failed"] for stats in category_stats.values())
    total_skipped = sum(stats["skipped"] for stats in category_stats.values())
    
    print(f"✅ SUCCESS!\n")
    print(f"📊 Statistics by Sheet:")
    for sheet, stats in category_stats.items():
        print(f"  • {sheet} (gid={stats['gid']}): {stats['embedded']}/{stats['total_rows']} embedded")
    
    print(f"\n📈 Total Summary:")
    print(f"  • Successfully embedded: {total_embedded}")
    print(f"  • Failed: {total_failed}")
    print(f"  • Skipped: {total_skipped}")
    print(f"\n📁 Files created:")
    print(f"  • all_embeddings.json ({len(all_embeddings)} embeddings)")
    print(f"  • products_metadata.json")
    print(f"  • embedding_stats.json")
    
    if total_embedded > 0:
        print(f"\n🚀 Ready to search! Run:")
        print(f"   streamlit run app_boq_search.py")
    else:
        print(f"\n⚠️  No products were embedded")

if __name__ == "__main__":
    main()