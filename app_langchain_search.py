import os
import re
import json
import math
import base64
from io import BytesIO
from collections import Counter
from typing import Any, Dict, List, Tuple

import clip
import torch
import numpy as np
import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from numpy import dot
from numpy.linalg import norm

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# ENV
# ============================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Design Discovery + LangChain",
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
div[data-testid="stTextInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

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

.dd-search-wrap {
    background: #fff;
    padding: 16px 56px 16px 56px;
    border-bottom: 1px solid #e8e8e8;
}
.dd-content { padding: 22px 56px 40px 56px; background: #f5f4f2; }
.dd-result-count { font-size: 0.8rem; color: #999; margin-bottom: 18px; }

.dd-chip-row {
    display:flex; gap:8px; flex-wrap:wrap; margin: 10px 0 16px 0;
}
.dd-chip {
    font-size: 0.72rem; background:#fff; border:1px solid #e4e4e4;
    border-radius:999px; padding:6px 10px; color:#444;
}

.dd-panel {
    background:#fff; border:1px solid #ebebeb; border-radius:12px;
    padding:16px; margin-bottom:16px;
}
.dd-panel-title {
    font-size:0.92rem; font-weight:700; color:#111; margin-bottom:10px;
}
.dd-panel-text {
    font-size:0.82rem; color:#666; line-height:1.5;
}

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
.dd-footer {
    display: flex; align-items: center; justify-content: space-between; margin-top: 7px;
}
.dd-score {
    font-size: 0.72rem; font-weight: 600; color: #6c5ce7;
    background: #f0eeff; border-radius: 50px; padding: 2px 8px;
}
.dd-open { font-size: 0.72rem; color: #aaa; text-decoration: none; }
.dd-open:hover { color: #555; text-decoration: underline; }

.dd-empty { text-align: center; padding: 80px 20px; }
.dd-empty-icon { font-size: 48px; margin-bottom: 14px; opacity: 0.35; }
.dd-empty-title { font-size: 1.25rem; font-weight: 700; color: #222; margin-bottom: 6px; }
.dd-empty-text { font-size: 0.9rem; color: #aaa; line-height: 1.6; }
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


@st.cache_resource
def get_llms():
    if not OPENAI_API_KEY:
        return None, None

    parser_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    reason_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=OPENAI_API_KEY
    )
    return parser_llm, reason_llm


embeddings_data = load_embeddings()
metadata = load_metadata()
model, preprocess, device = load_clip_model()
parser_llm, reason_llm = get_llms()


# ============================================================
# HELPERS
# ============================================================
def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na, nb = norm(a), norm(b)
    return float(dot(a, b) / (na * nb)) if na and nb else 0.0


def safe_meta(meta, key, default=""):
    v = meta.get(key, default)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text)


def tokenize_query(query):
    return [t for t in normalize_text(query).split() if len(t) > 1]


def build_searchable_text(meta):
    keys = [
        "project_name", "product_name", "project_id", "sku", "space_name",
        "category", "city", "country", "construction_type_text", "style",
        "designers", "render_created_by", "brand", "layout_id"
    ]
    return normalize_text(" ".join(safe_meta(meta, k) for k in keys))


@st.cache_resource
def build_metadata_index(_meta):
    rst, tdf, td = {}, Counter(), 0
    for key, meta_item in _meta.items():
        text = build_searchable_text(meta_item)
        rst[key] = text
        for t in set(tokenize_query(text)):
            tdf[t] += 1
        td += 1
    return rst, tdf, td


row_search_text, token_doc_freq, total_docs = build_metadata_index(metadata)


def metadata_score(query, stext):
    tokens = tokenize_query(query)
    if not tokens:
        return 0.0, [], []

    matched, missing, score, max_score = [], [], 0.0, 0.0
    for t in tokens:
        df = token_doc_freq.get(t, 0)
        w = math.log((total_docs + 1) / (df + 1)) + 1.0
        if df > 0:
            max_score += w
            if t in stext:
                score += w
                matched.append(t)
            else:
                missing.append(t)
    return (score / max_score if max_score else 0.0), matched, missing


def search_products_hybrid(
    query,
    top_k=20,
    cw=0.75,
    mw=0.25,
    min_clip=0.20,
    min_final=0.20,
    mratio=0.6,
    category_filter=""
):
    if not embeddings_data:
        return []

    ti = clip.tokenize([query]).to(device)
    with torch.no_grad():
        tf = model.encode_text(ti)
    qemb = tf.cpu().numpy().flatten().tolist()

    meta_terms = [t for t in tokenize_query(query) if token_doc_freq.get(t, 0) > 0]
    clip_terms = [t for t in tokenize_query(query) if token_doc_freq.get(t, 0) == 0]

    results = []
    category_filter_norm = normalize_text(category_filter) if category_filter else ""

    for pk, data in embeddings_data.items():
        emb = data.get("embedding")
        if not emb:
            continue

        meta_item = metadata.get(pk, {})
        row_category = normalize_text(
            safe_meta(meta_item, "space_name", safe_meta(meta_item, "category", ""))
        )

        if category_filter_norm and category_filter_norm not in row_category:
            continue

        st_text = row_search_text.get(pk, "")
        cs = cosine_similarity(qemb, emb)
        ms, matched, missing = metadata_score(query, st_text)

        if meta_terms:
            r = sum(1 for t in meta_terms if t in st_text) / max(len(meta_terms), 1)
            if r < mratio:
                continue

        if clip_terms and cs < min_clip:
            continue

        final = cw * cs + mw * ms
        if final < min_final:
            continue

        results.append((pk, final, cs, ms, matched, missing))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def fetch_b64(image_url):
    try:
        if not image_url or not str(image_url).startswith("http"):
            return None

        url = image_url.strip()
        if "drive.google.com" in url:
            if "/d/" in url:
                fid = url.split("/d/")[1].split("/")[0]
            elif "id=" in url:
                fid = url.split("id=")[1].split("&")[0]
            else:
                return None
            url = f"https://drive.google.com/uc?id={fid}&export=download"

        r = requests.get(url, timeout=15, allow_redirects=True)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.thumbnail((600, 400), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=82)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None
    return None


def esc(v):
    return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def trunc(s, n=24):
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


def extract_first_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {}


def compact_result_payload(results: List[Tuple]) -> List[Dict[str, Any]]:
    payload = []
    for pk, final, cs, ms, matched, missing in results[:10]:
        m = metadata.get(pk, {})
        payload.append({
            "key": pk,
            "project_id": safe_meta(m, "project_id", safe_meta(m, "sku", "")),
            "project_name": safe_meta(m, "project_name", safe_meta(m, "product_name", "")),
            "space_name": safe_meta(m, "space_name", safe_meta(m, "category", "")),
            "style": safe_meta(m, "construction_type_text", safe_meta(m, "style", "")),
            "city": safe_meta(m, "city", ""),
            "country": safe_meta(m, "country", ""),
            "layout_id": safe_meta(m, "layout_id", ""),
            "designer": safe_meta(m, "designers", safe_meta(m, "render_created_by", "")),
            "image_url": safe_meta(m, "image_url", ""),
            "render_url": safe_meta(m, "render_url", ""),
            "score": round(float(final), 4)
        })
    return payload


# ============================================================
# LANGCHAIN: QUERY PARSER
# ============================================================
QUERY_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an interior design search assistant.

Convert the user's raw search into strict JSON with these keys:
- cleaned_query: string
- category: string
- style: string
- tone: string
- space_type: string
- material_hint: string
- exclusions: array of strings
- should_recommend_related: boolean

Rules:
- Keep values short.
- If unknown, use empty string.
- Output ONLY valid JSON.
        """.strip(),
    ),
    ("human", "User query: {user_query}")
])


def parse_user_query(user_query: str) -> Dict[str, Any]:
    fallback = {
        "cleaned_query": user_query,
        "category": "",
        "style": "",
        "tone": "",
        "space_type": "",
        "material_hint": "",
        "exclusions": [],
        "should_recommend_related": False
    }

    if not parser_llm:
        return fallback

    try:
        chain = QUERY_PARSE_PROMPT | parser_llm
        response = chain.invoke({"user_query": user_query})
        parsed = extract_first_json(response.content)
        if not parsed:
            return fallback

        return {
            "cleaned_query": str(parsed.get("cleaned_query", user_query)).strip() or user_query,
            "category": str(parsed.get("category", "")).strip(),
            "style": str(parsed.get("style", "")).strip(),
            "tone": str(parsed.get("tone", "")).strip(),
            "space_type": str(parsed.get("space_type", "")).strip(),
            "material_hint": str(parsed.get("material_hint", "")).strip(),
            "exclusions": parsed.get("exclusions", []) if isinstance(parsed.get("exclusions", []), list) else [],
            "should_recommend_related": bool(parsed.get("should_recommend_related", False)),
        }
    except Exception:
        return fallback


def build_search_text(parsed: Dict[str, Any]) -> str:
    parts = [
        parsed.get("cleaned_query", ""),
        parsed.get("category", ""),
        parsed.get("style", ""),
        parsed.get("tone", ""),
        parsed.get("space_type", ""),
        parsed.get("material_hint", ""),
    ]
    parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return " ".join(dict.fromkeys(parts)) if parts else parsed.get("cleaned_query", "")


# ============================================================
# LANGCHAIN: RESULT REFINER
# ============================================================
RESULT_REFINE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an interior design recommendation assistant.

Given a user query, a parsed intent, and top search results,
pick the 3 best items and return strict JSON:

{
  "best_keys": ["key1", "key2", "key3"],
  "best_overall_key": "key1",
  "reason": "short reason",
  "grouping_label": "short label"
}

Rules:
- Use only keys present in the candidate list.
- Output ONLY valid JSON.
        """.strip(),
    ),
    (
        "human",
        "User query: {user_query}\nParsed intent: {parsed_intent}\nCandidates: {candidates}"
    )
])


def refine_results_with_llm(user_query: str, parsed: Dict[str, Any], raw_results: List[Tuple]) -> Dict[str, Any]:
    default = {
        "best_keys": [r[0] for r in raw_results[:3]],
        "best_overall_key": raw_results[0][0] if raw_results else "",
        "reason": "Best local similarity matches.",
        "grouping_label": "Top matches"
    }

    if not reason_llm or not raw_results:
        return default

    try:
        chain = RESULT_REFINE_PROMPT | reason_llm
        response = chain.invoke({
            "user_query": user_query,
            "parsed_intent": json.dumps(parsed, ensure_ascii=False),
            "candidates": json.dumps(compact_result_payload(raw_results), ensure_ascii=False)
        })
        refined = extract_first_json(response.content)
        if not refined:
            return default

        candidate_keys = {r[0] for r in raw_results}
        best_keys = [
            k for k in refined.get("best_keys", []) if isinstance(k, str) and k in candidate_keys
        ]
        if not best_keys:
            best_keys = default["best_keys"]

        best_overall_key = refined.get("best_overall_key", "")
        if best_overall_key not in candidate_keys:
            best_overall_key = default["best_overall_key"]

        return {
            "best_keys": best_keys[:3],
            "best_overall_key": best_overall_key,
            "reason": str(refined.get("reason", default["reason"])).strip(),
            "grouping_label": str(refined.get("grouping_label", default["grouping_label"])).strip()
        }
    except Exception:
        return default


# ============================================================
# LANGCHAIN: RELATED RECOMMENDATIONS
# ============================================================
RELATED_RECO_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an interior design compatibility assistant.

Return strict JSON:
{{
  "recommendations": [
    {{"category": "flooring", "query": "warm light oak flooring"}},
    {{"category": "table", "query": "minimal executive table light wood"}},
    {{"category": "wall_finish", "query": "neutral warm wall finish modern"}}
  ]
}}

Rules:
- Keep 2 to 4 recommendations.
- Output ONLY valid JSON.
        """
    ),
    (
        "human",
        "User query: {user_query}\nParsed intent: {parsed}\nSelected item: {selected_item}"
    )
])

def recommend_related_queries(user_query: str, parsed: Dict[str, Any], selected_key: str) -> List[Dict[str, str]]:
    if not reason_llm or not selected_key or selected_key not in metadata:
        return []

    selected_item = metadata[selected_key]
    selected_payload = {
        "project_id": safe_meta(selected_item, "project_id", safe_meta(selected_item, "sku", "")),
        "project_name": safe_meta(selected_item, "project_name", safe_meta(selected_item, "product_name", "")),
        "space_name": safe_meta(selected_item, "space_name", safe_meta(selected_item, "category", "")),
        "style": safe_meta(selected_item, "construction_type_text", safe_meta(selected_item, "style", "")),
        "city": safe_meta(selected_item, "city", ""),
        "country": safe_meta(selected_item, "country", ""),
        "layout_id": safe_meta(selected_item, "layout_id", ""),
        "designer": safe_meta(selected_item, "designers", safe_meta(selected_item, "render_created_by", "")),
    }

    try:
        chain = RELATED_RECO_PROMPT | reason_llm
        response = chain.invoke({
            "user_query": user_query,
            "parsed": json.dumps(parsed, ensure_ascii=False),
            "selected_item": json.dumps(selected_payload, ensure_ascii=False)
        })
        parsed_json = extract_first_json(response.content)
        recos = parsed_json.get("recommendations", [])
        cleaned = []
        for item in recos:
            if isinstance(item, dict):
                cat = str(item.get("category", "")).strip()
                qry = str(item.get("query", "")).strip()
                if cat and qry:
                    cleaned.append({"category": cat, "query": qry})
        return cleaned[:4]
    except Exception:
        return []


def run_related_searches(related_queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    output = []
    for item in related_queries:
        q = item["query"]
        cat = item["category"]
        found = search_products_hybrid(q, top_k=3, category_filter=cat, min_final=0.15, mratio=0.0)
        if not found:
            found = search_products_hybrid(q, top_k=3, min_final=0.15, mratio=0.0)

        best = []
        for pk, final, *_rest in found[:3]:
            m = metadata.get(pk, {})
            best.append({
                "key": pk,
                "name": safe_meta(m, "project_name", safe_meta(m, "product_name", "")),
                "space_name": safe_meta(m, "space_name", safe_meta(m, "category", "")),
                "style": safe_meta(m, "construction_type_text", safe_meta(m, "style", "")),
                "score": round(float(final), 4)
            })

        output.append({
            "category": cat,
            "query": q,
            "matches": best
        })

    return output


# ============================================================
# UI HELPERS
# ============================================================
def render_result_card(pk: str, final: float, refined_pick: bool = False, badge_text: str = ""):
    m = metadata.get(pk, {})
    image_url = safe_meta(m, "image_url")
    space = safe_meta(m, "space_name", safe_meta(m, "category", "—"))
    proj = safe_meta(m, "project_name", safe_meta(m, "product_name", ""))
    floor = safe_meta(m, "floor", safe_meta(m, "layout_id", ""))
    designer = safe_meta(m, "designers", safe_meta(m, "render_created_by", ""))
    style = safe_meta(m, "construction_type_text", safe_meta(m, "style", "Modern"))
    render_url = safe_meta(m, "render_url")

    b64 = fetch_b64(image_url)
    img_html = f'<img src="{b64}" alt="{esc(space)}" />' if b64 else '<div class="dd-no-img">No preview</div>'
    style_html = f'<div class="badge-style">{esc(trunc(style, 12))}</div>'
    meta1 = esc(trunc(proj, 22)) + (f" · {esc(floor)}" if floor and floor.lower() not in ("unknown", "") else "")
    meta2 = f'By <b>{esc(trunc(designer, 20))}</b>' if designer else ""
    open_btn = f'<a class="dd-open" href="{esc(render_url)}" target="_blank">Open ↗</a>' if render_url.startswith("http") else ""

    pick_marker = ""
    if refined_pick:
        label = badge_text or "AI Pick"
        pick_marker = f'<div class="badge-style" style="left:9px; right:auto; background:#eefaf1; color:#1f7a3e;">{esc(label)}</div>'

    st.markdown(f"""
<div class="dd-card">
  <div class="dd-card-img">
    {img_html}
    {style_html}
    {pick_marker}
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
</div>
""", unsafe_allow_html=True)


# ============================================================
# TOPBAR
# ============================================================
st.markdown("""
<div class="dd-topbar">
  <div class="dd-logo">⊞</div>
  <span class="dd-brand">Design Discovery + LangChain</span>
</div>
""", unsafe_allow_html=True)

if not embeddings_data:
    st.markdown("""
    <div class="dd-content">
      <div class="dd-empty">
        <div class="dd-empty-icon">⚠️</div>
        <div class="dd-empty-title">No Embeddings Found</div>
        <div class="dd-empty-text">Run your embedding-generation script first.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ============================================================
# SEARCH BAR
# ============================================================
st.markdown('<div class="dd-search-wrap">', unsafe_allow_html=True)
c1, c2 = st.columns([5, 1])
with c1:
    search_query = st.text_input(
        "search",
        placeholder="🔍 Search by style, room, material, city, or intent",
        label_visibility="collapsed"
    )
with c2:
    num_results = st.selectbox("n", [5, 10, 15, 20, 30, 50], index=1, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# CONTENT
# ============================================================
st.markdown('<div class="dd-content">', unsafe_allow_html=True)

if not OPENAI_API_KEY:
    st.markdown("""
    <div class="dd-panel">
      <div class="dd-panel-title">API key not found</div>
      <div class="dd-panel-text">
        Add <b>OPENAI_API_KEY</b> to your <b>.env</b> file to enable LangChain query parsing,
        reranking, and related recommendations. Local CLIP search still works.
      </div>
    </div>
    """, unsafe_allow_html=True)

if search_query:
    try:
        parsed = parse_user_query(search_query)
        optimized_query = build_search_text(parsed)
        category_filter = parsed.get("category", "")

        chips = []
        for key in ["category", "style", "tone", "space_type", "material_hint"]:
            val = parsed.get(key, "")
            if val:
                chips.append(f"<span class='dd-chip'>{esc(key)}: {esc(val)}</span>")

        st.markdown("""
        <div class="dd-panel">
          <div class="dd-panel-title">Parsed search intent</div>
          <div class="dd-panel-text">This is the start-side LangChain layer.</div>
          <div class="dd-chip-row">
        """ + "".join(chips) + """
          </div>
          <div class="dd-panel-text"><b>Optimized search text:</b> """ + esc(optimized_query) + """</div>
        </div>
        """, unsafe_allow_html=True)

        results = search_products_hybrid(
            optimized_query,
            top_k=num_results,
            cw=0.75,
            mw=0.25,
            min_clip=0.20,
            min_final=0.20,
            mratio=0.6,
            category_filter=category_filter
        )

        if not results and category_filter:
            results = search_products_hybrid(
                optimized_query,
                top_k=num_results,
                cw=0.75,
                mw=0.25,
                min_clip=0.20,
                min_final=0.20,
                mratio=0.4,
                category_filter=""
            )

        if not results:
            st.markdown("""
            <div class="dd-empty">
              <div class="dd-empty-icon">🔍</div>
              <div class="dd-empty-title">No results found</div>
              <div class="dd-empty-text">Try broader words like modern lounge, warm office, reception, breakout, executive</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            refined = refine_results_with_llm(search_query, parsed, results)
            best_keys = refined.get("best_keys", [])
            best_overall_key = refined.get("best_overall_key", "")
            ai_reason = refined.get("reason", "")
            grouping_label = refined.get("grouping_label", "Top matches")

            st.markdown(f"""
            <div class="dd-panel">
              <div class="dd-panel-title">AI refinement</div>
              <div class="dd-panel-text"><b>{esc(grouping_label)}</b></div>
              <div class="dd-panel-text" style="margin-top:8px;">{esc(ai_reason)}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<div class="dd-result-count">{len(results)} results</div>', unsafe_allow_html=True)

            cols = st.columns(5)
            for idx, (pk, final, cs, ms, matched, missing) in enumerate(results):
                with cols[idx % 5]:
                    refined_pick = pk in best_keys
                    badge_text = "Best overall" if pk == best_overall_key else ("AI pick" if refined_pick else "")
                    render_result_card(pk, final, refined_pick=refined_pick, badge_text=badge_text)

            should_related = parsed.get("should_recommend_related", False)
            if should_related and best_overall_key:
                related_queries = recommend_related_queries(search_query, parsed, best_overall_key)
                related_results = run_related_searches(related_queries) if related_queries else []

                if related_results:
                    st.markdown("""
                    <div class="dd-panel" style="margin-top:8px;">
                      <div class="dd-panel-title">Related recommendations</div>
                      <div class="dd-panel-text">This is the end-side LangChain layer generating linked follow-up searches.</div>
                    </div>
                    """, unsafe_allow_html=True)

                    for item in related_results:
                        st.markdown(f"""
                        <div class="dd-panel">
                          <div class="dd-panel-title">{esc(item['category'])}</div>
                          <div class="dd-panel-text"><b>Suggested query:</b> {esc(item['query'])}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if item["matches"]:
                            rel_cols = st.columns(min(3, len(item["matches"])))
                            for i, match in enumerate(item["matches"][:3]):
                                with rel_cols[i]:
                                    render_result_card(match["key"], match["score"], refined_pick=False)

    except Exception as e:
        st.error(f"Search error: {str(e)}")
else:
    st.markdown("""
    <div class="dd-empty">
      <div class="dd-empty-icon">✦</div>
      <div class="dd-empty-title">Search to discover renders</div>
      <div class="dd-empty-text">
        Try: premium Scandinavian chair for executive cabin · warm modern reception · minimal breakout lounge
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)