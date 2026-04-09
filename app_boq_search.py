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
import re
import html
import math
from collections import Counter

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Design Discovery",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS - CLEAN, MINIMAL, IMAGE-FOCUSED
# ============================================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #fafafa;
        color: #1a1a1a;
    }

    .main {
        background: #fafafa;
        padding: 0 !important;
    }

    /* Header/Search Section */
    .header-section {
        background: #ffffff;
        border-bottom: 1px solid #e8e8e8;
        padding: 24px 40px;
        position: sticky;
        top: 0;
        z-index: 100;
    }

    .header-inner {
        max-width: 1600px;
        margin: 0 auto;
    }

    .header-top {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
    }

    .header-icon {
        width: 32px;
        height: 32px;
        background: #f0f0f0;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }

    .header-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: 0.2px;
    }

    .search-wrapper {
        display: flex;
        align-items: center;
        position: relative;
        max-width: 500px;
    }

    .search-icon {
        position: absolute;
        left: 14px;
        color: #999;
        font-size: 16px;
        pointer-events: none;
    }

    .search-input {
        width: 100%;
        padding: 10px 14px 10px 40px;
        border: 1px solid #d8d8d8;
        border-radius: 8px;
        font-size: 0.95rem;
        background: #fff;
        color: #1a1a1a;
        transition: all 0.2s ease;
    }

    .search-input::placeholder {
        color: #999;
    }

    .search-input:focus {
        outline: none;
        border-color: #1a1a1a;
        box-shadow: 0 0 0 3px rgba(26, 26, 26, 0.05);
    }

    /* Results Grid - Image Focused */
    .results-section {
        max-width: 1600px;
        margin: 0 auto;
        padding: 32px 40px;
    }

    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 20px;
        animation: fadeIn 0.4s ease-out;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Card - Image Prominent */
    .design-card {
        background: #fff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e8e8e8;
        transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1);
        display: flex;
        flex-direction: column;
    }

    .design-card:hover {
        border-color: #ccc;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
        transform: translateY(-6px);
    }

    /* Image Container */
    .card-image-container {
        width: 100%;
        height: 280px;
        background: linear-gradient(135deg, #f0f0f0 0%, #e8e8e8 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        position: relative;
    }

    .card-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s cubic-bezier(0.23, 1, 0.32, 1);
    }

    .design-card:hover .card-image {
        transform: scale(1.08);
    }

    /* Status Badge - Top Left */
    .status-badge {
        position: absolute;
        top: 12px;
        left: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.95);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #1a1a1a;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background: #22c55e;
        border-radius: 50%;
    }

    /* Style Tag - Top Right */
    .style-tag {
        position: absolute;
        top: 12px;
        right: 12px;
        background: rgba(255, 255, 255, 0.95);
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #1a1a1a;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* Card Content */
    .card-content {
        padding: 16px;
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .card-space-type {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #666;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 6px;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .card-location {
        font-size: 0.8rem;
        color: #666;
        line-height: 1.4;
    }

    .card-designer {
        font-size: 0.8rem;
        color: #666;
        margin-top: 4px;
    }

    .card-designer strong {
        color: #1a1a1a;
        font-weight: 600;
    }

    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 100px 40px;
    }

    .empty-icon {
        font-size: 64px;
        margin-bottom: 16px;
        opacity: 0.5;
    }

    .empty-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 8px;
    }

    .empty-text {
        font-size: 0.95rem;
        color: #999;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #fafafa;
    }

    ::-webkit-scrollbar-thumb {
        background: #d0d0d0;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #999;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .results-grid {
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 16px;
        }

        .header-section {
            padding: 16px 20px;
        }

        .results-section {
            padding: 24px 20px;
        }

        .card-image-container {
            height: 200px;
        }

        .search-wrapper {
            max-width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOADERS
# ============================================================
@st.cache_resource
def load_embeddings():
    if not os.path.exists("all_embeddings.json"):
        return None
    with open("all_embeddings.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_metadata():
    if not os.path.exists("products_metadata.json"):
        return {}
    with open("products_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

@st.cache_resource
def build_metadata_index(metadata_dict):
    row_search_text = {}
    token_doc_freq = Counter()
    total_docs = 0

    for key, meta in metadata_dict.items():
        text = build_searchable_text(meta)
        row_search_text[key] = text

        tokens = set(tokenize_query(text))
        for token in tokens:
            token_doc_freq[token] += 1

        total_docs += 1

    return row_search_text, token_doc_freq, total_docs

embeddings_data = load_embeddings()
metadata = load_metadata()
model, preprocess, device = load_clip()

# ============================================================
# HELPERS
# ============================================================
def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    if norm(a) == 0 or norm(b) == 0:
        return 0.0

    return float(dot(a, b) / (norm(a) * norm(b)))

def safe_meta(meta, key, default=""):
    value = meta.get(key, default)
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_query(query):
    query = normalize_text(query)
    return [t for t in query.split() if len(t) > 1]

def build_searchable_text(meta):
    fields = [
        safe_meta(meta, "project_name"),
        safe_meta(meta, "product_name"),
        safe_meta(meta, "project_id"),
        safe_meta(meta, "sku"),
        safe_meta(meta, "space_name"),
        safe_meta(meta, "category"),
        safe_meta(meta, "city"),
        safe_meta(meta, "country"),
        safe_meta(meta, "construction_type_text"),
        safe_meta(meta, "style"),
        safe_meta(meta, "designers"),
        safe_meta(meta, "render_created_by"),
        safe_meta(meta, "brand"),
        safe_meta(meta, "layout_id"),
    ]
    return normalize_text(" ".join([f for f in fields if f]))

row_search_text, token_doc_freq, total_docs = build_metadata_index(metadata)

def split_query_by_corpus(query, token_doc_freq):
    tokens = tokenize_query(query)
    metadata_terms = []
    non_metadata_terms = []

    for token in tokens:
        if token_doc_freq.get(token, 0) > 0:
            metadata_terms.append(token)
        else:
            non_metadata_terms.append(token)

    return metadata_terms, non_metadata_terms

def compute_dynamic_metadata_score(query, searchable_text, token_doc_freq, total_docs):
    query_tokens = tokenize_query(query)
    if not query_tokens:
        return 0.0, [], []

    matched = []
    missing = []
    score = 0.0
    max_score = 0.0

    for token in query_tokens:
        df = token_doc_freq.get(token, 0)
        weight = math.log((total_docs + 1) / (df + 1)) + 1.0

        if df > 0:
            max_score += weight
            if token in searchable_text:
                score += weight
                matched.append(token)
            else:
                missing.append(token)

    if max_score == 0:
        return 0.0, matched, missing

    return score / max_score, matched, missing

def search_products_hybrid(
    query,
    top_k=20,
    clip_weight=0.75,
    metadata_weight=0.25,
    min_clip_score=0.20,
    min_final_score=0.20,
    metadata_match_ratio=0.6
):
    if not embeddings_data:
        return []

    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)

    query_emb = text_features.cpu().numpy().flatten().tolist()
    metadata_terms, non_metadata_terms = split_query_by_corpus(query, token_doc_freq)

    results = []

    for product_key, data in embeddings_data.items():
        emb = data.get("embedding")
        if not emb:
            continue

        meta = metadata.get(product_key, {})
        searchable_text = row_search_text.get(product_key, "")

        clip_score = cosine_similarity(query_emb, emb)
        metadata_score, matched_terms, missing_terms = compute_dynamic_metadata_score(
            query, searchable_text, token_doc_freq, total_docs
        )

        if metadata_terms:
            matched_metadata_terms = [t for t in metadata_terms if t in searchable_text]
            ratio = len(matched_metadata_terms) / len(metadata_terms)
            if ratio < metadata_match_ratio:
                continue

        if non_metadata_terms and clip_score < min_clip_score:
            continue

        final_score = (clip_weight * clip_score) + (metadata_weight * metadata_score)

        if final_score < min_final_score:
            continue

        results.append((
            product_key,
            final_score,
            clip_score,
            metadata_score,
            matched_terms,
            missing_terms
        ))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def download_image_from_url(image_url):
    try:
        if not image_url or not isinstance(image_url, str):
            return None

        image_url = image_url.strip()
        if not image_url.startswith("http"):
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
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image
    except:
        pass

    return None

def esc(value):
    return html.escape(str(value), quote=True)

# ============================================================
# UI - HEADER & SEARCH
# ============================================================
st.markdown("""
<div class="header-section">
    <div class="header-inner">
        <div class="header-top">
            <div class="header-icon">📐</div>
            <div class="header-title">Design Discovery</div>
        </div>
        <div class="search-wrapper">
            <span class="search-icon">🔍</span>
""", unsafe_allow_html=True)

if not embeddings_data:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⚠️</div>
        <div class="empty-title">No Data Found</div>
        <div class="empty-text">Please run the processing script first to generate embeddings and metadata files.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

search_query = st.text_input(
    "search",
    placeholder="Search By Space",
    label_visibility="collapsed",
    key="search_input"
)

st.markdown("""
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# RESULTS GRID
# ============================================================
st.markdown("""<div class="results-section">""", unsafe_allow_html=True)

if search_query:
    try:
        all_results = search_products_hybrid(
            search_query,
            top_k=30,
            clip_weight=0.75,
            metadata_weight=0.25,
            min_clip_score=0.20,
            min_final_score=0.20,
            metadata_match_ratio=0.6
        )

        if not all_results:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🔍</div>
                <div class="empty-title">No Results</div>
                <div class="empty-text">Try searching with different keywords like space type, location, or style.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""<div class="results-grid">""", unsafe_allow_html=True)

            for product_key, final_score, clip_score, metadata_score, matched_terms, missing_terms in all_results:
                prod_meta = metadata.get(product_key, {})

                image_url = safe_meta(prod_meta, "image_url", "")
                project_name = safe_meta(prod_meta, "project_name", safe_meta(prod_meta, "product_name", "Unknown"))
                space_name = safe_meta(prod_meta, "space_name", safe_meta(prod_meta, "category", "Unknown"))
                city = safe_meta(prod_meta, "city", "Unknown")
                country = safe_meta(prod_meta, "country", "Unknown")
                designer = safe_meta(prod_meta, "designers", "Unknown")
                construction_type = safe_meta(prod_meta, "construction_type_text", safe_meta(prod_meta, "style", "Modern"))

                st.markdown("""<div class="design-card">""", unsafe_allow_html=True)

                # Image with badges
                st.markdown("""<div class="card-image-container">""", unsafe_allow_html=True)

                if image_url:
                    image = download_image_from_url(image_url)
                    if image:
                        st.image(image, use_column_width=True)

                st.markdown(f"""
                    <div class="status-badge">
                        <div class="status-dot"></div>
                        Approved
                    </div>
                    <div class="style-tag">{esc(construction_type)}</div>
                </div>
                """, unsafe_allow_html=True)

                # Content
                st.markdown(f"""
                <div class="card-content">
                    <div>
                        <div class="card-space-type">{esc(space_name)}</div>
                        <div class="card-title">{esc(project_name)}</div>
                        <div class="card-location">
                            {esc(city)} • {esc(country)}
                        </div>
                        <div class="card-designer">
                            By <strong>{esc(designer)}</strong>
                        </div>
                    </div>
                </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""</div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Search error: {str(e)}")

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">✨</div>
        <div class="empty-title">Explore Global Designs</div>
        <div class="empty-text">Search for inspiration by space type, location, style, or designer name.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""</div>""", unsafe_allow_html=True)