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

# Page configuration
st.set_page_config(
    page_title="Smart Product Finder",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 0 0 20px 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .header-container h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .search-container {
        max-width: 800px;
        margin: 0 auto 3rem;
        display: flex;
        gap: 1rem;
        align-items: center;
        flex-wrap: wrap;
        justify-content: center;
    }
    
    .category-section {
        margin-bottom: 3rem;
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #667eea;
    }
    
    .category-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .category-title::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 1.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
        margin-right: 0.3rem;
    }
    
    .products-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1.5rem;
    }
    
    .product-card {
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
    }
    
    .product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    
    .product-image {
        width: 100%;
        height: 200px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    
    .product-info {
        padding: 1.2rem;
    }
    
    .product-name {
        font-size: 1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.3rem;
        line-height: 1.3;
        min-height: 2.6rem;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    
    .product-brand {
        font-size: 0.75rem;
        color: #667eea;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .product-meta {
        font-size: 0.8rem;
        color: #718096;
        margin-bottom: 0.8rem;
        line-height: 1.5;
    }
    
    .product-meta strong {
        color: #4a5568;
        font-weight: 600;
    }
    
    .match-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .no-results {
        text-align: center;
        padding: 3rem 2rem;
        color: #718096;
    }
    
    .search-stats {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .stat-box {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #718096;
        margin-top: 0.3rem;
    }
    
    .empty-state {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 4rem 2rem;
        text-align: center;
        color: #718096;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)

# Load embeddings
@st.cache_resource
def load_embeddings():
    if not os.path.exists("all_embeddings.json"):
        return None
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

def search_products(query_embedding, top_k=100):
    """Search products by embedding similarity"""
    if not embeddings_data:
        return []
    
    similarities = {}
    
    for product_key, data in embeddings_data.items():
        sim = cosine_similarity(query_embedding, data["embedding"])
        similarities[product_key] = sim
    
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

def download_image_from_url(image_url):
    """Download image from any URL (S3, Google Drive, HTTP, etc.)"""
    try:
        if not image_url or not isinstance(image_url, str):
            return None
        
        # Handle Google Drive URLs
        if "drive.google.com" in image_url:
            # Extract file ID
            if "/d/" in image_url:
                file_id = image_url.split("/d/")[1].split("/")[0]
            elif "id=" in image_url:
                file_id = image_url.split("id=")[1].split("&")[0]
            else:
                return None
            
            # Construct download URL
            image_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        # Download image with timeout
        response = requests.get(image_url, timeout=15, allow_redirects=True)
        
        if response.status_code == 200:
            # Open as PIL image
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
    except:
        pass
    
    return None

def get_categories():
    """Get all unique categories from metadata"""
    categories = {}
    for product_key, meta in metadata.items():
        cat = meta.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(product_key)
    return categories

# ============================================================
# MAIN UI
# ============================================================

# Header
st.markdown("""
<div class="header-container">
    <h1>🛍️ Smart Product Finder</h1>
    <p>Search your product catalog using AI-powered visual search</p>
</div>
""", unsafe_allow_html=True)

# Check if embeddings exist
if not embeddings_data:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">⚠️</div>
        <h2>No embeddings found</h2>
        <p>Please run <code>embed_from_boq_sheets.py</code> first to generate embeddings from your Google Sheets.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Search interface
col1, col2 = st.columns([4, 1])

with col1:
    search_query = st.text_input(
        "Search",
        placeholder="e.g., 'blue chair', 'modern table', 'grey storage'...",
        label_visibility="collapsed"
    )

with col2:
    num_results = st.number_input("Results per category", min_value=3, max_value=12, value=5, label_visibility="collapsed")

# Search button
if st.button("🔍 Search", use_container_width=True, key="search_btn"):
    if search_query.strip():
        # Embed search query
        text_input = clip.tokenize([search_query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
        query_emb = text_features.cpu().numpy().flatten().tolist()
        
        # Search all products
        all_results = search_products(query_emb, top_k=len(metadata))
        
        # Organize results by category
        results_by_category = {}
        categories = get_categories()
        
        for cat in categories:
            results_by_category[cat] = []
        
        for product_key, similarity in all_results:
            prod_meta = metadata.get(product_key, {})
            cat = prod_meta.get("category", "Unknown")
            
            if cat in results_by_category and len(results_by_category[cat]) < num_results:
                results_by_category[cat].append((product_key, similarity))
        
        # Count total results
        total_results = sum(len(v) for v in results_by_category.values())
        
        if total_results == 0:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🔍</div>
                <h2>No results found</h2>
                <p>Try searching with different keywords or check your data.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Search stats
            st.markdown(f"""
            <div class="search-stats">
                <div class="stat-box">
                    <div class="stat-value">{total_results}</div>
                    <div class="stat-label">Total Matches</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len([c for c in results_by_category.values() if c])}</div>
                    <div class="stat-label">Categories</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Display results by category
            for category in get_categories().keys():
                results = results_by_category.get(category, [])
                
                if results:
                    st.markdown(f"""
                    <div class="category-section">
                        <div class="category-title">{category}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    cols = st.columns(5)
                    
                    for idx, (product_key, similarity) in enumerate(results):
                        col = cols[idx % 5]
                        
                        with col:
                            prod_meta = metadata.get(product_key, {})
                            image_url = prod_meta.get("image_url", "")
                            product_name = prod_meta.get("product_name", "Unknown")
                            sku = prod_meta.get("sku", "Unknown")
                            color = prod_meta.get("color", "Unknown")
                            brand = prod_meta.get("brand", "Unknown")
                            style = prod_meta.get("style", "Unknown")
                            
                            # Product card container
                            st.markdown("""
                            <div class="product-card">
                            """, unsafe_allow_html=True)
                            
                            # Image
                            if image_url:
                                image = download_image_from_url(image_url)
                                if image:
                                    st.image(image, use_column_width=True)
                                else:
                                    st.markdown("""
                                    <div class="product-image" style="display: flex; align-items: center; justify-content: center; color: #a0aec0; font-size: 0.9rem;">
                                        <span>Image unavailable</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Product info
                            st.markdown(f"""
                            <div class="product-info">
                                <div class="product-brand">{brand if brand != 'Unknown' else ''}</div>
                                <div class="product-name">{product_name}</div>
                                <div class="product-meta">
                                    <strong>SKU:</strong> {sku[:15]}<br>
                                    <strong>Color:</strong> {color if color != 'Unknown' else 'N/A'}<br>
                                    {f"<strong>Style:</strong> {style}<br>" if style != 'Unknown' else ""}
                                </div>
                                <div class="match-badge">{similarity:.0%} Match</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
else:
    # Default state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">👆</div>
        <h2>Enter a search query to get started</h2>
        <p>Search for colors, styles, brands, or any product description</p>
    </div>
    """, unsafe_allow_html=True)