import pandas as pd
import json
import numpy as np
from PIL import Image
import clip
import torch
import os
import requests
from io import BytesIO

# Load CLIP model
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Using device: {device}\n")

# ============================================================
# STEP 1: Load data from Google Sheets or CSV
# ============================================================

def load_data_from_sheets(csv_url_or_path):
    """
    Load data from:
    - CSV file: "data.csv"
    - CSV URL: "https://example.com/data.csv"
    - Google Sheets: "https://docs.google.com/spreadsheets/d/13uC1teuK_Iej0Lax1rFEWpDzwpPQKH9lJhjDkrNNJd4/edit?gid=722702960#gid=722702960"
    """
    print("Loading data from source...")
    
    if csv_url_or_path.startswith("http"):
        # URL - download it
        try:
            df = pd.read_csv(csv_url_or_path)
        except Exception as e:
            print(f"Error: Could not read from URL: {csv_url_or_path}")
            print(f"Details: {str(e)}")
            print("Make sure the Google Sheets export URL is correct")
            exit()
    else:
        # Local CSV file
        if not os.path.exists(csv_url_or_path):
            print(f"Error: File not found: {csv_url_or_path}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in current directory:")
            for f in os.listdir():
                print(f"  - {f}")
            exit()
        df = pd.read_csv(csv_url_or_path)
    
    print(f"✓ Loaded {len(df)} rows\n")
    print("Columns found:")
    for col in df.columns:
        print(f"  • {col}")
    print()
    
    return df

# ============================================================
# STEP 2: Download image from Google Drive
# ============================================================

def extract_drive_id(drive_url):
    """Extract file ID from Google Drive URL"""
    try:
        # Handle different Drive URL formats
        if "/d/" in drive_url:
            file_id = drive_url.split("/d/")[1].split("/")[0]
        elif "id=" in drive_url:
            file_id = drive_url.split("id=")[1].split("&")[0]
        else:
            return None
        return file_id
    except:
        return None

def download_image_from_drive(drive_url):
    """Download image from Google Drive URL and return PIL Image"""
    try:
        # Extract file ID
        file_id = extract_drive_id(drive_url)
        
        if not file_id:
            print(f"  ✗ Invalid Drive URL format")
            return None
        
        # Construct download URL (this is the key - direct download)
        download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        # Download image with timeout
        response = requests.get(download_url, timeout=15, allow_redirects=True)
        
        if response.status_code == 200:
            # Open as PIL image
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        else:
            print(f"  ✗ Download failed (status {response.status_code})")
            return None
            
    except requests.exceptions.Timeout:
        print(f"  ✗ Download timeout (image too large or slow connection)")
        return None
    except Exception as e:
        print(f"  ✗ Error downloading: {str(e)[:50]}")
        return None

# ============================================================
# STEP 3: Embed image
# ============================================================

def embed_image(image):
    """Convert image to embedding using CLIP"""
    try:
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        return image_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"  ✗ Error embedding: {str(e)[:50]}")
        return None

# ============================================================
# STEP 4: Main process
# ============================================================

def main():
    # Configure these settings
    
    # ===== CHANGE THIS TO YOUR DATA SOURCE =====
    DATA_SOURCE = "data.csv"  # Change to your CSV file or Google Sheets export URL
    # ============================================
    
    # Column names in your sheet (adjust to match your actual columns)
    IMAGE_URL_COLUMN = "Thumbnail Drive Link"  # Column with Google Drive links
    PRODUCT_NAME_COLUMN = "Name"  # Product name
    SKU_COLUMN = "SKU"  # Product SKU
    CATEGORY_COLUMN = "Category"  # Product category
    SUB_CATEGORY_COLUMN = "Sub-Category"  # Sub-category
    
    print("=" * 60)
    print("EMBEDDING FROM GOOGLE SHEETS/CSV")
    print("=" * 60)
    print()
    
    # Load data
    df = load_data_from_sheets(DATA_SOURCE)
    
    # Check required columns exist
    required_columns = [IMAGE_URL_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        exit()
    
    # Create embeddings dictionary
    all_embeddings = {}
    metadata = {}
    successful = 0
    failed = 0
    
    print(f"Embedding {len(df)} products...\n")
    
    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] ", end="", flush=True)
        
        # Get drive URL
        drive_url = row[IMAGE_URL_COLUMN]
        if pd.isna(drive_url) or not str(drive_url).startswith("http"):
            print(f"✗ Invalid URL")
            failed += 1
            continue
        
        # Get product info
        product_name = str(row.get(PRODUCT_NAME_COLUMN, "Unknown"))
        sku = str(row.get(SKU_COLUMN, "Unknown"))
        category = str(row.get(CATEGORY_COLUMN, "Unknown"))
        sub_category = str(row.get(SUB_CATEGORY_COLUMN, "Unknown"))
        
        # Create unique key
        unique_key = f"{sku}_{product_name[:20]}"
        
        print(f"Downloading {sku[:15]}...", end=" ", flush=True)
        
        # Download image
        image = download_image_from_drive(drive_url)
        if image is None:
            print("✗ Failed")
            failed += 1
            continue
        
        print("✓ Embedding...", end=" ", flush=True)
        
        # Embed image
        embedding = embed_image(image)
        if embedding is None:
            print("✗ Failed")
            failed += 1
            continue
        
        # Store embedding and metadata
        all_embeddings[unique_key] = {
            "embedding": embedding
        }
        
        metadata[unique_key] = {
            "sku": sku,
            "product_name": product_name,
            "category": category,
            "sub_category": sub_category,
            "drive_url": drive_url,
            "row_index": idx
        }
        
        print("✓ Done")
        successful += 1
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    with open("all_embeddings.json", "w") as f:
        json.dump(all_embeddings, f)
    
    with open("products_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ SUCCESS!")
    print(f"  Total products: {len(df)}")
    print(f"  Embedded successfully: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nFiles created:")
    print(f"  • all_embeddings.json ({len(all_embeddings)} embeddings)")
    print(f"  • products_metadata.json (product info)")
    
    if successful > 0:
        print("\n✓ Ready to search! Run:")
        print("  streamlit run app_sheets_search.py")
    else:
        print("\n⚠ No products were embedded. Check your data source and Drive links.")

if __name__ == "__main__":
    main()