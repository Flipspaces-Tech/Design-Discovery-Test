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

# Custom CSS - Beautiful Modern Design
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [class*="css"] {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        background: #0f1419;
        color: #e0e0e0;
    }
    
    .main {
        background: #0f1419;
    }
    
    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 40px;
        border-radius: 20px;
        margin-bottom: 40px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 400px;
        height: 400px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        z-index: 0;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    
    .hero p {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 300;
    }
    
    /* Search Section */
    .search-section {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        padding: 40px;
        border-radius: 20px;
        margin-bottom: 50px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .search-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 20px;
        color: #fff;
    }
    
    .search-input-wrapper {
        position: relative;
    }
    
    .search-input-wrapper input {
        width: 100%;
        padding: 18px 20px !important;
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        color: #fff !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .search-input-wrapper input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .search-input-wrapper input:focus {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.15) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .search-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 40px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        margin-top: 15px !important;
    }
    
    .search-button:hover {
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Results Stats */
    .results-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 40px 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-color: #667eea;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.15);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .stat-label {
        font-size: 0.95rem;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Category Section */
    .category-section {
        margin-bottom: 50px;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .category-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .category-icon {
        width: 10px;
        height: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    .category-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #fff;
        letter-spacing: -0.5px;
    }
    
    /* Product Grid */
    .products-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 25px;
        margin-bottom: 40px;
    }
    
    .product-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        cursor: pointer;
        position: relative;
    }
    
    .product-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
        z-index: 1;
    }
    
    .product-card:hover::before {
        left: 100%;
    }
    
    .product-card:hover {
        transform: translateY(-10px);
        border-color: #667eea;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .product-image-wrapper {
        width: 100%;
        height: 220px;
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        position: relative;
    }
    
    .product-image-wrapper img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    
    .product-card:hover .product-image-wrapper img {
        transform: scale(1.08);
    }
    
    .product-info {
        padding: 20px;
    }
    
    .product-brand {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #667eea;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .product-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 10px;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .product-details {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 12px;
        line-height: 1.6;
    }
    
    .product-details strong {
        color: #aaa;
    }
    
    .match-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-align: center;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 80px 40px;
    }
    
    .empty-icon {
        font-size: 80px;
        margin-bottom: 20px;
        opacity: 0.5;
    }
    
    .empty-title {
        font-size: 2rem;
        font-weight: 800;
        color: #fff;
        margin-bottom: 10px;
    }
    
    .empty-text {
        font-size: 1.1rem;
        color: #888;
    }
    
    /* No Results */
    .no-results-container {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border-radius: 20px;
        padding: 60px 40px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 40px 0;
    }
    
    .no-results-icon {
        font-size: 60px;
        margin-bottom: 20px;
        opacity: 0.6;
    }
    
    .no-results-text {
        color: #888;
        font-size: 1.1rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #16213e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
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
    if not embeddings_data:
        return []
    
    similarities = {}
    for product_key, data in embeddings_data.items():
        sim = cosine_similarity(query_embedding, data["embedding"])
        similarities[product_key] = sim
    
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

def download_image_from_url(image_url):
    try:
        if not image_url or not isinstance(image_url, str):
            return None
        
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

def get_categories():
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

# Hero Section
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <h1>🛍️ Smart Product Finder</h1>
        <p>Discover products using AI-powered intelligent search</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Check if embeddings exist
if not embeddings_data:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⚠️</div>
        <div class="empty-title">No Embeddings Found</div>
        <div class="empty-text">Please run the processing script first</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Search Section
st.markdown("""<div class="search-section">""", unsafe_allow_html=True)
st.markdown("""<div class="search-title">🔍 What are you looking for?</div>""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    search_query = st.text_input(
        "search",
        placeholder="Search by color, style, brand, or product name...",
        label_visibility="collapsed"
    )

with col2:
    num_results = st.selectbox(
        "results",
        [3, 5, 8, 10, 12],
        index=1,
        label_visibility="collapsed",
        key="num_results"
    )

st.markdown("""</div>""", unsafe_allow_html=True)

# Search execution
if search_query:
    try:
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
        active_categories = len([c for c in results_by_category.values() if c])
        
        if total_results == 0:
            st.markdown("""
            <div class="no-results-container">
                <div class="no-results-icon">🔍</div>
                <div class="no-results-text">No products found matching your search</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Results Stats
            st.markdown("""<div class="results-stats">""", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{total_results}</div>
                    <div class="stat-label">Total Results</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{active_categories}</div>
                    <div class="stat-label">Categories</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""</div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display results by category
            for category in sorted(get_categories().keys()):
                results = results_by_category.get(category, [])
                
                if results:
                    st.markdown(f"""
                    <div class="category-section">
                        <div class="category-header">
                            <div class="category-icon"></div>
                            <div class="category-title">{category}</div>
                        </div>
                        <div class="products-grid">
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
                            
                            # Product card
                            st.markdown("""<div class="product-card">""", unsafe_allow_html=True)
                            
                            # Image
                            st.markdown("""<div class="product-image-wrapper">""", unsafe_allow_html=True)
                            if image_url:
                                image = download_image_from_url(image_url)
                                if image:
                                    st.image(image, use_column_width=True)
                                else:
                                    st.markdown("<span style='color: #666;'>No image</span>", unsafe_allow_html=True)
                            st.markdown("""</div>""", unsafe_allow_html=True)
                            
                            # Info
                            st.markdown(f"""
                            <div class="product-info">
                                <div class="product-brand">{brand if brand != 'Unknown' else ''}</div>
                                <div class="product-name">{product_name}</div>
                                <div class="product-details">
                                    <strong>SKU:</strong> {sku[:12]}<br>
                                    <strong>Color:</strong> {color if color != 'Unknown' else 'N/A'}
                                </div>
                                <div class="match-badge">{similarity:.0%} Match</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""</div>""", unsafe_allow_html=True)
                    
                    st.markdown("""
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Search error: {str(e)}")

else:
    # Default state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">✨</div>
        <div class="empty-title">Ready to Search</div>
        <div class="empty-text">Enter a search term above to explore our product catalog</div>
    </div>
    """, unsafe_allow_html=True)