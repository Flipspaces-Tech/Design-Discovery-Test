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
# CUSTOM CSS — matches the "Design Discovery" screenshot UI
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    *, *::before, *::after {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background: #f8f7f5;
        color: #1a1a1a;
    }

    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ── TOP NAV ── */
    .topbar {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 32px;
        background: #fff;
        border-bottom: 1px solid #ebebeb;
        position: sticky;
        top: 0;
        z-index: 100;
    }

    .logo-mark {
        width: 34px;
        height: 34px;
        background: #111;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .logo-mark svg {
        width: 18px;
        height: 18px;
        fill: #fff;
    }

    .brand-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111;
        letter-spacing: -0.3px;
    }

    /* ── SEARCH BAR ── */
    .search-wrapper {
        padding: 20px 32px 0 32px;
        background: #fff;
    }

    .search-inner {
        position: relative;
        max-width: 480px;
    }

    .search-icon {
        position: absolute;
        left: 14px;
        top: 50%;
        transform: translateY(-50%);
        color: #999;
        font-size: 14px;
        pointer-events: none;
    }

    /* Override Streamlit input */
    .search-wrapper .stTextInput > div > div > input {
        background: #f5f5f5 !important;
        border: 1px solid #e8e8e8 !important;
        border-radius: 50px !important;
        padding: 10px 16px 10px 40px !important;
        font-size: 0.9rem !important;
        color: #333 !important;
        box-shadow: none !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .search-wrapper .stTextInput > div > div > input:focus {
        border-color: #c5c5c5 !important;
        background: #fff !important;
        box-shadow: 0 0 0 3px rgba(0,0,0,0.06) !important;
    }

    .search-wrapper .stTextInput label { display: none !important; }

    /* ── FILTER BAR ── */
    .filter-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 14px 32px 16px 32px;
        background: #fff;
        border-bottom: 1px solid #ebebeb;
        flex-wrap: wrap;
    }

    .filter-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 6px 14px;
        border: 1px solid #ddd;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 500;
        color: #333;
        background: #fff;
        cursor: pointer;
        transition: all 0.15s ease;
        white-space: nowrap;
    }

    .filter-chip:hover {
        border-color: #aaa;
        background: #f9f9f9;
    }

    .filter-chip .chevron {
        font-size: 0.65rem;
        color: #888;
        margin-left: 2px;
    }

    .filter-chip-more {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border: 1px solid #ddd;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 500;
        color: #555;
        background: #fff;
        cursor: pointer;
        margin-left: auto;
    }

    /* ── MAIN CONTENT AREA ── */
    .content-area {
        padding: 28px 32px;
        background: #f8f7f5;
    }

    /* ── RESULT COUNT ── */
    .result-count {
        font-size: 0.82rem;
        color: #888;
        margin-bottom: 20px;
        font-weight: 400;
    }

    /* ── PRODUCT GRID ── */
    /* Using Streamlit columns — we just style the cards */

    .dd-card {
        background: #fff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #ebebeb;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
        margin-bottom: 20px;
        cursor: pointer;
    }

    .dd-card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.10);
        transform: translateY(-3px);
    }

    .dd-img-wrapper {
        width: 100%;
        height: 185px;
        overflow: hidden;
        background: #f0f0f0;
        position: relative;
    }

    .dd-img-wrapper img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.35s ease;
    }

    .dd-card:hover .dd-img-wrapper img {
        transform: scale(1.04);
    }

    /* Status badge */
    .badge-approved {
        position: absolute;
        top: 10px;
        left: 10px;
        background: #fff;
        color: #1a9e5a;
        border: 1px solid #c3eed8;
        border-radius: 50px;
        padding: 3px 9px;
        font-size: 0.7rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
        backdrop-filter: blur(4px);
    }

    .badge-approved::before {
        content: '✓';
        font-size: 0.65rem;
    }

    /* Style tag */
    .badge-style {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(255,255,255,0.92);
        color: #444;
        border-radius: 50px;
        padding: 3px 9px;
        font-size: 0.7rem;
        font-weight: 500;
        backdrop-filter: blur(4px);
    }

    /* Card body */
    .dd-card-body {
        padding: 12px 14px 14px;
    }

    .dd-space-name {
        font-size: 0.92rem;
        font-weight: 600;
        color: #111;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .dd-meta-row {
        font-size: 0.78rem;
        color: #888;
        display: flex;
        align-items: center;
        gap: 4px;
        margin-bottom: 1px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .dd-meta-row .dot {
        width: 3px;
        height: 3px;
        border-radius: 50%;
        background: #ccc;
        flex-shrink: 0;
    }

    .dd-meta-row a {
        color: #888;
        text-decoration: none;
    }

    .dd-meta-row a:hover {
        color: #333;
        text-decoration: underline;
    }

    /* ── EMPTY / NO RESULTS ── */
    .empty-state {
        text-align: center;
        padding: 100px 40px;
    }

    .empty-icon {
        font-size: 56px;
        margin-bottom: 16px;
        opacity: 0.4;
    }

    .empty-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #222;
        margin-bottom: 6px;
    }

    .empty-text {
        font-size: 0.95rem;
        color: #999;
    }

    /* ── HIDE STREAMLIT CHROME ── */
    #MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }
    .stSelectbox label { display: none !important; }

    /* selectbox styling */
    .stSelectbox > div > div {
        border-radius: 50px !important;
        border-color: #ddd !important;
        background: #fff !important;
        font-size: 0.82rem !important;
        min-height: 36px !important;
        font-family: 'DM Sans', sans-serif !important;
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
    metadata_terms, non_metadata_terms = [], []
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
    matched, missing = [], []
    score = max_score = 0.0
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

def search_products_hybrid(query, top_k=20, clip_weight=0.75, metadata_weight=0.25,
                            min_clip_score=0.20, min_final_score=0.20, metadata_match_ratio=0.6):
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
            query, searchable_text, token_doc_freq, total_docs)
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
        results.append((product_key, final_score, clip_score, metadata_score, matched_terms, missing_terms))
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
            return Image.open(BytesIO(response.content)).convert("RGB")
    except:
        pass
    return None

def esc(value):
    return html.escape(str(value), quote=True)

def truncate(text, max_len=22):
    return text if len(text) <= max_len else text[:max_len] + "…"

# ============================================================
# TOP NAV
# ============================================================
st.markdown("""
<div class="topbar">
    <div class="logo-mark">
        <svg viewBox="0 0 24 24"><path d="M3 3h8v8H3zm10 0h8v8h-8zM3 13h8v8H3zm10 4h2v-2h2v2h2v2h-2v2h-2v-2h-2z"/></svg>
    </div>
    <span class="brand-name">Design Discovery</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SEARCH + FILTER BAR
# ============================================================
if not embeddings_data:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⚠️</div>
        <div class="empty-title">No Embeddings Found</div>
        <div class="empty-text">Please run the processing script first.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="search-wrapper">', unsafe_allow_html=True)
col_s, col_r = st.columns([4, 1])
with col_s:
    # Inject search icon overlay via markdown
    st.markdown('<span class="search-icon">🔍</span>', unsafe_allow_html=True)
    search_query = st.text_input(
        "search",
        placeholder="Search By Space",
        label_visibility="collapsed"
    )
with col_r:
    num_results = st.selectbox(
        "results",
        [5, 10, 15, 20, 30, 50],
        index=1,
        label_visibility="collapsed"
    )
st.markdown('</div>', unsafe_allow_html=True)

# Filter chips (decorative — wired to search logic via query)
st.markdown("""
<div class="filter-bar">
    <div class="filter-chip">Construction Type <span class="chevron">▾</span></div>
    <div class="filter-chip">Project Type <span class="chevron">▾</span></div>
    <div class="filter-chip">Project Name <span class="chevron">▾</span></div>
    <div class="filter-chip">Space Name <span class="chevron">▾</span></div>
    <div class="filter-chip">Designer <span class="chevron">▾</span></div>
    <div class="filter-chip">Country <span class="chevron">▾</span></div>
    <div class="filter-chip-more">⚙ More</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# RESULTS
# ============================================================
st.markdown('<div class="content-area">', unsafe_allow_html=True)

if search_query:
    try:
        all_results = search_products_hybrid(
            search_query,
            top_k=num_results,
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
                <div class="empty-title">No results found</div>
                <div class="empty-text">Try a different combination of space type, designer, city, or style.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-count">{len(all_results)} results</div>', unsafe_allow_html=True)

            cols = st.columns(5)
            for idx, (product_key, final_score, clip_score, metadata_score, matched_terms, missing_terms) in enumerate(all_results):
                with cols[idx % 5]:
                    prod_meta = metadata.get(product_key, {})

                    image_url         = safe_meta(prod_meta, "image_url", "")
                    space_name        = safe_meta(prod_meta, "space_name",   safe_meta(prod_meta, "category", "Unknown"))
                    project_name      = safe_meta(prod_meta, "project_name", safe_meta(prod_meta, "product_name", ""))
                    floor_level       = safe_meta(prod_meta, "floor", safe_meta(prod_meta, "layout_id", ""))
                    designer          = safe_meta(prod_meta, "designers",    safe_meta(prod_meta, "render_created_by", ""))
                    construction_type = safe_meta(prod_meta, "construction_type_text", safe_meta(prod_meta, "style", "Modern"))
                    render_url        = safe_meta(prod_meta, "render_url", "")
                    is_approved       = safe_meta(prod_meta, "approved", "").lower() in ("true", "1", "yes", "approved")

                    # Card open tag
                    st.markdown('<div class="dd-card">', unsafe_allow_html=True)

                    # Image area
                    st.markdown('<div class="dd-img-wrapper">', unsafe_allow_html=True)
                    if image_url:
                        image = download_image_from_url(image_url)
                        if image:
                            st.image(image, use_column_width=True)
                        else:
                            st.markdown("<div style='height:185px;display:flex;align-items:center;justify-content:center;color:#bbb;font-size:0.8rem;'>No preview</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='height:185px;display:flex;align-items:center;justify-content:center;color:#bbb;font-size:0.8rem;'>No preview</div>", unsafe_allow_html=True)

                    # Overlay badges
                    approved_badge = '<div class="badge-approved">Approved</div>' if is_approved else ''
                    st.markdown(f"""
                        {approved_badge}
                        <div class="badge-style">{esc(construction_type)}</div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)  # /dd-img-wrapper

                    # Card body
                    floor_str = f" • {esc(floor_level)}" if floor_level and floor_level.lower() not in ("unknown", "") else ""
                    proj_str = truncate(project_name, 20) if project_name else ""
                    by_str = truncate(designer, 18) if designer else ""

                    open_html = ""
                    if render_url and render_url.startswith("http"):
                        open_html = f'<a class="dd-meta-row" href="{esc(render_url)}" target="_blank" style="color:#666;text-decoration:underline;font-size:0.75rem;">Open ↗</a>'

                    st.markdown(f"""
                    <div class="dd-card-body">
                        <div class="dd-space-name">{esc(space_name)}</div>
                        <div class="dd-meta-row">
                            <span>{esc(proj_str)}</span>
                            {f'<span class="dot"></span><span>{floor_str.lstrip(" • ")}</span>' if floor_str else ''}
                        </div>
                        <div class="dd-meta-row">
                            {f'<span>By {esc(by_str)}</span>' if by_str else ''}
                            <span class="dot"></span>
                            <span style="background:linear-gradient(90deg,#6c5ce7,#a29bfe);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:600;">{final_score:.0%}</span>
                        </div>
                        {open_html}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)  # /dd-card

    except Exception as e:
        st.error(f"Search error: {str(e)}")

else:
    # Show placeholder grid with empty cards to mimic the browse state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">✦</div>
        <div class="empty-title">Search to discover renders</div>
        <div class="empty-text">Try: conference room Hyderabad &nbsp;·&nbsp; warm modern lounge &nbsp;·&nbsp; reception ground floor</div>
    </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /content-area