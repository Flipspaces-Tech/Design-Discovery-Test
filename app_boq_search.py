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
import html as html_lib
import math
import base64
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
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: #f5f4f2 !important;
    color: #1a1a1a;
}

.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}

section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* TOPBAR */
.dd-topbar {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 56px;
    background: #ffffff;
    border-bottom: 1px solid #e8e8e8;
}
.dd-logo {
    width: 32px; height: 32px; background: #111;
    border-radius: 7px; display: flex; align-items: center;
    justify-content: center; font-size: 15px; color: #fff; flex-shrink: 0;
}
.dd-brand { font-size: 1rem; font-weight: 700; color: #111; letter-spacing: -0.2px; }

/* Streamlit input override */
div[data-testid="stTextInput"] label { display: none !important; }
div[data-testid="stTextInput"] > div > div > input {
    border-radius: 50px !important;
    border: 1px solid #e0e0e0 !important;
    background: #ffffff !important;
    padding: 9px 16px 9px 36px !important;
    font-size: 0.875rem !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #222 !important;
    box-shadow: none !important;
    caret-color: #222 !important;
}
div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #bbb !important;
    background: #fff !important;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.07) !important;
}
/* Remove dark container around input */
div[data-testid="stTextInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* selectbox override */
div[data-testid="stSelectbox"] label { display: none !important; }
div[data-testid="stSelectbox"] > div > div {
    border-radius: 50px !important;
    border: 1px solid #e0e0e0 !important;
    background: #ffffff !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    min-height: 38px !important;
    color: #333 !important;
}
div[data-testid="stSelectbox"] > div {
    background: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stSelectbox"] {
    background: #ffffff !important;
}
/* The actual visible select element */
div[data-testid="stSelectbox"] > div > div > div {
    background: #ffffff !important;
    color: #333 !important;
}
/* Dropdown popup */
div[data-testid="stSelectbox"] ul {
    background: #fff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 12px !important;
}
div[data-testid="stSelectbox"] li {
    color: #333 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* CONTENT */
.dd-content { padding: 22px 56px 40px 56px; background: #f5f4f2; }
.dd-result-count { font-size: 0.8rem; color: #999; margin-bottom: 18px; }

/* CARDS — entirely self-contained HTML, no st.image() */
.dd-card {
    background: #fff; border-radius: 12px; overflow: hidden;
    border: 1px solid #ebebeb; margin-bottom: 20px;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.dd-card:hover {
    box-shadow: 0 8px 28px rgba(0,0,0,0.11);
    transform: translateY(-3px);
}
.dd-card-img {
    width: 100%; height: 180px; position: relative;
    overflow: hidden; background: #e8e8e8;
}
.dd-card-img img {
    width: 100%; height: 100%; object-fit: cover; display: block;
    transition: transform 0.35s ease;
}
.dd-card:hover .dd-card-img img { transform: scale(1.05); }
.dd-no-img {
    width: 100%; height: 180px; background: #efefef;
    display: flex; align-items: center; justify-content: center;
    color: #bbb; font-size: 0.78rem;
}
.badge-approved {
    position: absolute; top: 9px; left: 9px;
    background: #fff; color: #18a060; border: 1px solid #b8edcd;
    border-radius: 50px; padding: 3px 8px;
    font-size: 0.68rem; font-weight: 600;
}
.badge-style {
    position: absolute; top: 9px; right: 9px;
    background: rgba(255,255,255,0.88); color: #444;
    border-radius: 50px; padding: 3px 8px; font-size: 0.68rem; font-weight: 500;
}
.dd-card-body { padding: 11px 13px 13px; }
.dd-space {
    font-size: 0.9rem; font-weight: 600; color: #111; margin-bottom: 3px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.dd-meta {
    font-size: 0.76rem; color: #999; margin-bottom: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.dd-meta b { color: #666; font-weight: 500; }
.dd-footer { display: flex; align-items: center; justify-content: space-between; margin-top: 7px; }
.dd-score {
    font-size: 0.72rem; font-weight: 600; color: #6c5ce7;
    background: #f0eeff; border-radius: 50px; padding: 2px 8px;
}
.dd-open { font-size: 0.72rem; color: #aaa; text-decoration: none; }
.dd-open:hover { color: #555; text-decoration: underline; }

/* EMPTY */
.dd-empty { text-align: center; padding: 80px 20px; }
.dd-empty-icon { font-size: 48px; margin-bottom: 14px; opacity: 0.35; }
.dd-empty-title { font-size: 1.25rem; font-weight: 700; color: #222; margin-bottom: 6px; }
.dd-empty-text { font-size: 0.9rem; color: #aaa; line-height: 1.6; }

/* Search wrapper */
.dd-search-wrap {
    background: #fff;
    padding: 16px 56px 16px 56px;
    border-bottom: 1px solid #e8e8e8;
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
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl, preprocess = clip.load("ViT-B/32", device=device)
    return mdl, preprocess, device

embeddings_data = load_embeddings()
metadata = load_metadata()
model, preprocess, device = load_clip_model()

# ============================================================
# HELPERS
# ============================================================
def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32); b = np.array(b, dtype=np.float32)
    na, nb = norm(a), norm(b)
    return float(dot(a, b) / (na * nb)) if na and nb else 0.0

def safe_meta(meta, key, default=""):
    v = meta.get(key, default)
    if v is None: return default
    v = str(v).strip(); return v if v else default

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text)

def tokenize_query(query):
    return [t for t in normalize_text(query).split() if len(t) > 1]

def build_searchable_text(meta):
    keys = ["project_name","product_name","project_id","sku","space_name",
            "category","city","country","construction_type_text","style",
            "designers","render_created_by","brand","layout_id"]
    return normalize_text(" ".join(safe_meta(meta, k) for k in keys))

@st.cache_resource
def build_metadata_index(_meta):
    rst, tdf, td = {}, Counter(), 0
    for key, meta in _meta.items():
        text = build_searchable_text(meta)
        rst[key] = text
        for t in set(tokenize_query(text)): tdf[t] += 1
        td += 1
    return rst, tdf, td

row_search_text, token_doc_freq, total_docs = build_metadata_index(metadata)

def metadata_score(query, stext):
    tokens = tokenize_query(query)
    if not tokens: return 0.0, [], []
    matched, missing, score, max_score = [], [], 0.0, 0.0
    for t in tokens:
        df = token_doc_freq.get(t, 0)
        w = math.log((total_docs + 1) / (df + 1)) + 1.0
        if df > 0:
            max_score += w
            if t in stext: score += w; matched.append(t)
            else: missing.append(t)
    return (score / max_score if max_score else 0.0), matched, missing

def search_products_hybrid(query, top_k=20, cw=0.75, mw=0.25,
                            min_clip=0.20, min_final=0.20, mratio=0.6):
    if not embeddings_data: return []
    ti = clip.tokenize([query]).to(device)
    with torch.no_grad():
        tf = model.encode_text(ti)
    qemb = tf.cpu().numpy().flatten().tolist()
    meta_terms = [t for t in tokenize_query(query) if token_doc_freq.get(t, 0) > 0]
    clip_terms = [t for t in tokenize_query(query) if token_doc_freq.get(t, 0) == 0]
    results = []
    for pk, data in embeddings_data.items():
        emb = data.get("embedding")
        if not emb: continue
        st_text = row_search_text.get(pk, "")
        cs = cosine_similarity(qemb, emb)
        ms, matched, missing = metadata_score(query, st_text)
        if meta_terms:
            r = sum(1 for t in meta_terms if t in st_text) / len(meta_terms)
            if r < mratio: continue
        if clip_terms and cs < min_clip: continue
        final = cw * cs + mw * ms
        if final < min_final: continue
        results.append((pk, final, cs, ms, matched, missing))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def fetch_b64(image_url):
    try:
        if not image_url or not str(image_url).startswith("http"): return None
        url = image_url.strip()
        if "drive.google.com" in url:
            if "/d/" in url: fid = url.split("/d/")[1].split("/")[0]
            elif "id=" in url: fid = url.split("id=")[1].split("&")[0]
            else: return None
            url = f"https://drive.google.com/uc?id={fid}&export=download"
        r = requests.get(url, timeout=15, allow_redirects=True)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.thumbnail((600, 400), Image.LANCZOS)
            buf = BytesIO(); img.save(buf, format="JPEG", quality=82)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except: pass
    return None

def esc(v): return html_lib.escape(str(v), quote=True)
def trunc(s, n=24): return s if len(s) <= n else s[:n] + "…"

# ============================================================
# TOPBAR
# ============================================================
st.markdown("""
<div class="dd-topbar">
  <div class="dd-logo">⊞</div>
  <span class="dd-brand">Design Discovery</span>
</div>
""", unsafe_allow_html=True)

if not embeddings_data:
    st.markdown("""<div class="dd-content"><div class="dd-empty">
      <div class="dd-empty-icon">⚠️</div>
      <div class="dd-empty-title">No Embeddings Found</div>
      <div class="dd-empty-text">Please run the processing script first.</div>
    </div></div>""", unsafe_allow_html=True)
    st.stop()

# ============================================================
# SEARCH BAR (white background block)
# ============================================================
st.markdown('<div class="dd-search-wrap">', unsafe_allow_html=True)
c1, c2 = st.columns([5, 1])
with c1:
    search_query = st.text_input("search", placeholder="🔍  Search By Space", label_visibility="collapsed")
with c2:
    num_results = st.selectbox("n", [5, 10, 15, 20, 30, 50], index=1, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# RESULTS
# ============================================================
st.markdown('<div class="dd-content">', unsafe_allow_html=True)

if search_query:
    try:
        results = search_products_hybrid(
            search_query, top_k=num_results,
            cw=0.75, mw=0.25, min_clip=0.20, min_final=0.20, mratio=0.6
        )
        if not results:
            st.markdown("""<div class="dd-empty">
              <div class="dd-empty-icon">🔍</div>
              <div class="dd-empty-title">No results found</div>
              <div class="dd-empty-text">Try: conference room Hyderabad &nbsp;·&nbsp; warm modern lounge</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="dd-result-count">{len(results)} results</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            for idx, (pk, final, cs, ms, matched, missing) in enumerate(results):
                with cols[idx % 5]:
                    m = metadata.get(pk, {})
                    image_url  = safe_meta(m, "image_url")
                    space      = safe_meta(m, "space_name",   safe_meta(m, "category", "—"))
                    proj       = safe_meta(m, "project_name", safe_meta(m, "product_name", ""))
                    floor      = safe_meta(m, "floor",        safe_meta(m, "layout_id", ""))
                    designer   = safe_meta(m, "designers",    safe_meta(m, "render_created_by", ""))
                    style      = safe_meta(m, "construction_type_text", safe_meta(m, "style", "Modern"))
                    render_url = safe_meta(m, "render_url")
                    approved   = safe_meta(m, "approved", "").lower() in ("true","1","yes","approved")

                    b64 = fetch_b64(image_url)
                    img_html = f'<img src="{b64}" alt="{esc(space)}" />' if b64 else \
                               f'<div class="dd-no-img">No preview</div>'

                    appr_html  = '<div class="badge-approved">✓ Approved</div>' if approved else ''
                    style_html = f'<div class="badge-style">{esc(trunc(style, 12))}</div>'

                    meta1 = esc(trunc(proj, 22)) + (f" · {esc(floor)}" if floor and floor.lower() not in ("unknown","") else "")
                    meta2 = f'By <b>{esc(trunc(designer, 20))}</b>' if designer else ""

                    open_btn = ""
                    if render_url and render_url.startswith("http"):
                        open_btn = f'<a class="dd-open" href="{esc(render_url)}" target="_blank">Open ↗</a>'

                    st.markdown(f"""
<div class="dd-card">
  <div class="dd-card-img">
    {img_html}
    {appr_html}
    {style_html}
  </div>
  <div class="dd-card-body">
    <div class="dd-space">{esc(space)}</div>
    <div class="dd-meta">{meta1}</div>
    <div class="dd-meta">{meta2}</div>
    <div class="dd-footer">
      <span class="dd-score">{final:.0%} match</span>
      {open_btn}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Search error: {str(e)}")
else:
    st.markdown("""<div class="dd-empty">
      <div class="dd-empty-icon">✦</div>
      <div class="dd-empty-title">Search to discover renders</div>
      <div class="dd-empty-text">
        Try: conference room Hyderabad &nbsp;·&nbsp; warm modern lounge &nbsp;·&nbsp; reception ground floor
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)