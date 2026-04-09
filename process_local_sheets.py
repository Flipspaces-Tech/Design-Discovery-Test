import pandas as pd
import json
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
# HELPER FUNCTIONS
# ============================================================

def download_image_from_url(image_url):
    """Download image from any URL (S3, Google Drive, HTTP, etc.) and return PIL Image"""
    try:
        if pd.isna(image_url) or not isinstance(image_url, str):
            return None

        image_url = str(image_url).strip()
        if not image_url.startswith("http"):
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

        response = requests.get(image_url, timeout=20, allow_redirects=True)

        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
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
        if images_field.startswith("["):
            try:
                parsed = json.loads(images_field)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "url" in item:
                            urls.append(item["url"])
                        elif isinstance(item, str) and item.startswith("http"):
                            urls.append(item)
            except:
                pass

        # Try comma-separated values
        if not urls and "," in images_field:
            parts = images_field.split(",")
            for part in parts:
                url = part.strip()
                if url.startswith("http"):
                    urls.append(url)

        # Single URL
        if not urls and images_field.startswith("http"):
            urls.append(images_field)

        return urls

    except:
        return []


def safe_str(val, default="Unknown"):
    if pd.isna(val):
        return default
    text = str(val).strip()
    return text if text else default


# ============================================================
# MAIN PROCESS
# ============================================================

def main():
    print("=" * 70)
    print("LOCAL CSV PROCESSOR - EMBEDDING GENERATION")
    print("=" * 70)
    print()

    # Get CSV folder
    print("Step 1: Specify the folder containing CSV files")
    print("-" * 70)
    csv_folder = input("📁 CSV folder name [default: DOWNLOAD_SHEET]: ").strip()
    if not csv_folder:
        csv_folder = "DOWNLOAD_SHEET"

    # Check if folder exists
    if not os.path.isdir(csv_folder):
        print(f"\n❌ Folder '{csv_folder}' not found")
        return

    # Find all CSV files
    csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])

    if not csv_files:
        print(f"\n❌ No CSV files found in '{csv_folder}'")
        return

    print(f"\n✓ Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        filepath = os.path.join(csv_folder, csv_file)
        size = os.path.getsize(filepath)
        print(f"   • {csv_file} ({size} bytes)")

    print(f"\n✓ Using Design Render Repository structure:")
    print(f"   • Images column: 'images'")
    print(f"   • Search name: 'project_name'")
    print(f"   • Unique ID: 'project_id'")
    print(f"   • Category: 'space_name'")
    print(f"   • Type/Style: 'construction_type_text'")
    print(f"   • Extra metadata: city, country, layout_id, designers, render_created_by, render_url")

    print(f"\n{'=' * 70}")
    print("PROCESSING CSV FILES")
    print(f"{'=' * 70}\n")

    all_embeddings = {}
    metadata = {}
    category_stats = {}

    for file_idx, csv_file in enumerate(csv_files, 1):
        filepath = os.path.join(csv_folder, csv_file)
        sheet_name = csv_file.replace(".csv", "")

        print(f"\n[{file_idx}/{len(csv_files)}] Processing: {csv_file}")
        print("-" * 70)

        try:
            print("  Loading CSV...", end=" ", flush=True)
            df = pd.read_csv(filepath)
            print(f"✓ {len(df)} rows found")
            cols = list(df.columns)
            print(f"  Columns: {', '.join(cols[:8])}... ({len(cols)} total)")
        except Exception as e:
            print(f"❌ Error: {str(e)[:100]}")
            continue

        if "images" not in df.columns:
            print("  ⚠️ 'images' column not found")
            print("     Skipping this file...")
            continue

        successful = 0
        failed = 0
        skipped = 0

        for idx, row in df.iterrows():
            project_name = safe_str(row.get("project_name"), "")
            project_id = safe_str(row.get("project_id"), "")

            # Skip only if these are missing
            if not project_name or not project_id:
                skipped += 1
                continue

            print(f"  [{idx + 1}/{len(df)}] ", end="", flush=True)

            image_urls = extract_image_urls_from_field(row.get("images"))

            if not image_urls:
                print("✗ no image")
                failed += 1
                continue

            primary_image_url = image_urls[0]

            category = safe_str(row.get("space_name"), "Unknown Space")
            style = safe_str(row.get("construction_type_text"), "Unknown")
            city = safe_str(row.get("city"), "Unknown")
            country = safe_str(row.get("country"), "Unknown")
            carpet_area = safe_str(row.get("project_carpet_area"), "Unknown")
            unit = safe_str(row.get("unit"), "Unknown")
            layout_id = safe_str(row.get("layout_id"), "Unknown")
            designers = safe_str(row.get("designers"), "Unknown")
            render_created_by = safe_str(row.get("render_created_by"), "Unknown")
            render_url = safe_str(row.get("render_url"), "")

            unique_key = f"{category}_{project_id}_{project_name[:25]}"

            print(f"{project_id[:12]}...", end=" ", flush=True)

            image = download_image_from_url(primary_image_url)
            if image is None:
                print("✗ image download")
                failed += 1
                continue

            print("✓", end=" ", flush=True)

            embedding = embed_image(image)
            if embedding is None:
                print("✗ embedding")
                failed += 1
                continue

            # Keep structure compatible with your current search app
            all_embeddings[unique_key] = {
                "embedding": embedding,
                "category": category
            }

            metadata[unique_key] = {
                # Required by your current Streamlit app
                "sku": project_id,
                "product_name": project_name,
                "category": category,
                "color": "N/A",
                "style": style,
                "brand": render_created_by,

                # Actual repository metadata
                "project_id": project_id,
                "project_name": project_name,
                "city": city,
                "country": country,
                "construction_type_text": style,
                "space_name": category,
                "project_carpet_area": carpet_area,
                "unit": unit,
                "layout_id": layout_id,
                "designers": designers,
                "render_created_by": render_created_by,
                "image_url": primary_image_url,
                "all_image_urls": image_urls,
                "render_url": render_url,
                "row_index": idx,
                "source_file": csv_file
            }

            print("✓")
            successful += 1

        category_stats[sheet_name] = {
            "file": csv_file,
            "total_rows": len(df),
            "embedded": successful,
            "failed": failed,
            "skipped": skipped
        }

        print(f"\n  ✓ Summary: {successful} embedded, {failed} failed, {skipped} skipped")

    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print(f"{'=' * 70}\n")

    with open("all_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(all_embeddings, f)

    with open("products_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open("embedding_stats.json", "w", encoding="utf-8") as f:
        json.dump(category_stats, f, indent=2, ensure_ascii=False)

    total_embedded = sum(stats["embedded"] for stats in category_stats.values())
    total_failed = sum(stats["failed"] for stats in category_stats.values())
    total_skipped = sum(stats["skipped"] for stats in category_stats.values())

    print("✅ SUCCESS!\n")
    print("📊 Statistics by File:")
    for sheet, stats in category_stats.items():
        print(f"  • {stats['file']}: {stats['embedded']}/{stats['total_rows']} embedded")

    print("\n📈 Total Summary:")
    print(f"  • Successfully embedded: {total_embedded}")
    print(f"  • Failed: {total_failed}")
    print(f"  • Skipped: {total_skipped}")

    print("\n📁 Files created:")
    print(f"  • all_embeddings.json ({len(all_embeddings)} embeddings)")
    print("  • products_metadata.json")
    print("  • embedding_stats.json")

    if total_embedded > 0:
        print("\n🚀 Ready to search! Run:")
        print("   streamlit run app_boq_search.py")
    else:
        print("\n⚠️ No renders were embedded")


if __name__ == "__main__":
    main()