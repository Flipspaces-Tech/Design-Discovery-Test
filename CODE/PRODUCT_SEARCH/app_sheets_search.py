import streamlit as st
from PIL import Image
import json
import clip
import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
import requests
from io import BytesIO

st.set_page_config(page_title="Product Finder", layout="wide")
st.title("🛍️ Smart Product Finder")
st.subheader("Search products from your catalog using AI")

# Load embeddings
@st.cache_resource
def load_embeddings():
    if not os.path.exists("all_embeddings.json"):
        st.error("❌ all_embeddings.json not found! Run embed_from_sheets.py first.")
        st.stop()
    with open("all_embeddings.json", "r") as f:
        return json.load(f)

# Load metadata
@st.cache_resource
def load_metadata():
    if not os.path.exists("products_metadata.json"):
        return {}
    with open("products_metadata.json", "r") as f:
        return json.load(f)

# Load CLIP
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

embeddings_data = load_embeddings()
metadata = load_metadata()
model, preprocess, device = load_clip()

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def search_products(query_embedding, top_k=12):
    """Search products by embedding similarity"""
    similarities = {}
    
    for product_key, data in embeddings_data.items():
        sim = cosine_similarity(query_embedding, data["embedding"])
        similarities[product_key] = sim
    
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

def download_image_from_drive(drive_url):
    """Download image from Google Drive"""
    try:
        # Extract file ID
        if "/d/" in drive_url:
            file_id = drive_url.split("/d/")[1].split("/")[0]
        elif "id=" in drive_url:
            file_id = drive_url.split("id=")[1].split("&")[0]
        else:
            return None
        
        # Download
        download_url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(download_url, timeout=10)
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
    except:
        pass
    
    return None

# Get categories for filter
def get_categories():
    categories = set()
    for product_key, meta in metadata.items():
        cat = meta.get("category", "Unknown")
        if cat != "Unknown":
            categories.add(cat)
    return sorted(list(categories))

categories = get_categories()

# Sidebar
st.sidebar.title("🔍 Search Options")
search_type = st.sidebar.radio("How do you want to search?", 
                                ["Custom Text Search", "Upload Image", "Browse by Category"])

# Category filter
if categories:
    st.sidebar.write("**Filter by category (optional):**")
    category_options = ["All Categories"] + categories
    selected_category = st.sidebar.selectbox("Category", category_options, label_visibility="collapsed")
else:
    selected_category = "All Categories"

results = []
search_query = ""

# ============================================================
# CUSTOM TEXT SEARCH
# ============================================================
if search_type == "Custom Text Search":
    st.sidebar.write("**Describe what you're looking for:**")
    custom_text = st.sidebar.text_area(
        "Enter your search",
        placeholder="e.g., 'modern wooden chair with leather' or 'spacious sofa for office'",
        height=100,
        label_visibility="collapsed"
    )
    
    num_results = st.sidebar.slider("Number of results", 5, 30, 12)
    
    if st.sidebar.button("🔍 Search", use_container_width=True):
        if custom_text.strip():
            # Embed search text
            text_input = clip.tokenize([custom_text]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_input)
            query_emb = text_features.cpu().numpy().flatten().tolist()
            
            # Search
            raw_results = search_products(query_emb, top_k=num_results * 2)
            
            # Filter by category if selected
            if selected_category != "All Categories":
                results = [
                    (key, sim) for key, sim in raw_results
                    if metadata[key].get("category") == selected_category
                ][:num_results]
            else:
                results = raw_results[:num_results]
            
            search_query = custom_text
            st.success(f"Found {len(results)} matching products!")
        else:
            st.warning("Please enter a search query!")

# ============================================================
# UPLOAD IMAGE SEARCH
# ============================================================
elif search_type == "Upload Image":
    st.sidebar.write("**Upload a product image:**")
    uploaded = st.sidebar.file_uploader("Choose image", type=['jpg', 'png', 'jpeg'], 
                                        label_visibility="collapsed")
    
    num_results = st.sidebar.slider("Number of results", 5, 30, 12)
    
    if uploaded is not None:
        image = Image.open(uploaded).convert('RGB')
        st.sidebar.image(image, caption="Your image", width=150)
        
        if st.sidebar.button("🔍 Find Similar", use_container_width=True):
            # Embed image
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
            query_emb = image_features.cpu().numpy().flatten().tolist()
            
            # Search
            raw_results = search_products(query_emb, top_k=num_results * 2)
            
            # Filter by category if selected
            if selected_category != "All Categories":
                results = [
                    (key, sim) for key, sim in raw_results
                    if metadata[key].get("category") == selected_category
                ][:num_results]
            else:
                results = raw_results[:num_results]
            
            search_query = "Uploaded image"
            st.success(f"Found {len(results)} similar products!")

# ============================================================
# BROWSE BY CATEGORY
# ============================================================
else:  # Browse by Category
    st.sidebar.write("**Select a category:**")
    
    if categories:
        browse_category = st.sidebar.selectbox("Category", categories, label_visibility="collapsed")
        
        if st.sidebar.button("📂 Browse", use_container_width=True):
            # Get all products in this category
            category_products = [
                (key, 1.0) for key, meta in metadata.items()
                if meta.get("category") == browse_category
            ]
            results = sorted(category_products, key=lambda x: x[0])[:30]
            search_query = f"Category: {browse_category}"
            st.success(f"Found {len(results)} products in {browse_category}")
    else:
        st.warning("No categories available")

# ============================================================
# DISPLAY RESULTS
# ============================================================
if results:
    st.write("---")
    st.subheader(f"📸 Results for: {search_query}")
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Results", len(results))
    with col2:
        best_match = results[0][1] if results else 0
        st.metric("Best Match", f"{best_match:.1%}")
    with col3:
        st.metric("Results Shown", len(results))
    
    st.write("---")
    
    # Display grid
    cols = st.columns(4)
    
    for idx, (product_key, similarity) in enumerate(results):
        col = cols[idx % 4]
        
        with col:
            # Get metadata
            prod_meta = metadata.get(product_key, {})
            drive_url = prod_meta.get("drive_url", "")
            product_name = prod_meta.get("product_name", "Unknown")
            sku = prod_meta.get("sku", "Unknown")
            category = prod_meta.get("category", "Unknown")
            sub_category = prod_meta.get("sub_category", "Unknown")
            
            # Try to display image
            if drive_url:
                image = download_image_from_drive(drive_url)
                if image:
                    st.image(image, use_column_width=True)
                else:
                    st.write("❌ Image unavailable")
            
            # Display product info
            st.write(f"**{product_name}**")
            st.write(f"SKU: `{sku}`")
            st.write(f"Category: {category}")
            if sub_category and sub_category != "Unknown":
                st.write(f"Type: {sub_category}")
            st.write(f"Match: **{similarity:.1%}**")
            
            st.divider()

else:
    st.info("👈 Use the sidebar to search for products")