import streamlit as st
from PIL import Image
import json
import clip
import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os

# Page config
st.set_page_config(page_title="Wooden Texture Finder", layout="wide")
st.title("🌳 Wooden Texture Finder")
st.subheader("Find textures that match your design style")

# Load embeddings
@st.cache_resource
def load_embeddings():
    with open("embeddings.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

embeddings_data = load_embeddings()
model, preprocess, device = load_clip_model()

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def search_textures(query_embedding, top_k=12):
    similarities = {}
    for image_name, data in embeddings_data.items():
        sim = cosine_similarity(query_embedding, data["embedding"])
        similarities[image_name] = sim
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# Sidebar: Choose input method
st.sidebar.title("Search Options")
search_method = st.sidebar.radio("How do you want to search?", 
                                  ["By Design Style", "By Image Upload"])

results = []

if search_method == "By Design Style":
    st.sidebar.subheader("Pick a design style:")
    
    styles = {
        "Scandinavian": "light wood, natural finish, minimalist, warm, matte, oak, pine",
        "Industrial": "dark metal, rough, glossy, walnut, steel, raw, bold",
        "Minimalist": "light, clean, simple, matte, neutral, pure, oak",
        "Boho": "warm, varied, natural, textured, mixed, earthy, honey",
        "Modern": "sleek, smooth, dark, glossy, contemporary, refined, ebony",
        "Rustic": "rough, warm, natural, aged, weathered, reclaimed, pine",
        "Luxury": "dark, glossy, refined, walnut, mahogany, polished, premium"
    }
    
    selected_style = st.sidebar.selectbox("Choose style:", list(styles.keys()))
    
    if st.sidebar.button("🔍 Search"):
        description = styles[selected_style]
        
        # Embed description
        text_input = clip.tokenize([description]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
        
        query_embedding = text_features.cpu().numpy().flatten().tolist()
        results = search_textures(query_embedding, top_k=12)
        
        st.success(f"Found {len(results)} textures matching '{selected_style}' style!")

else:  # Image upload
    st.sidebar.subheader("Upload inspiration image:")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Show uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.sidebar.image(image, caption="Your inspiration", width=200)
        
        if st.sidebar.button("🔍 Find Similar Textures"):
            # Embed uploaded image
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
            
            query_embedding = image_features.cpu().numpy().flatten().tolist()
            results = search_textures(query_embedding, top_k=12)
            
            st.success(f"Found {len(results)} similar textures!")

# Display results
if results:
    st.write("---")
    st.subheader("Top Matches:")
    
    # Create grid
    cols = st.columns(4)
    
    for idx, (image_name, similarity) in enumerate(results):
        col_idx = idx % 4
        
        with cols[col_idx]:
            try:
                # Load and display image
                image_path = os.path.join("./textures", image_name)
                image = Image.open(image_path)
                st.image(image, caption=f"{image_name}\nSimilarity: {similarity:.2f}")
            except:
                st.write(f"❌ Could not load {image_name}")

else:
    if search_method == "By Design Style":
        st.info("👈 Select a style and click Search to find textures")
    else:
        st.info("👈 Upload an image and click Search to find similar textures")