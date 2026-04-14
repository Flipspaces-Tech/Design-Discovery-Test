import os
import json
import numpy as np
from PIL import Image
import clip
import torch

# Load CLIP model (downloads automatically, ~350MB)
print("Loading CLIP model (first time takes ~1 min)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to your texture images
image_folder = "./textures"  # Change this to your folder

# Create embeddings
embeddings_data = {}
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

print(f"\nFound {len(image_files)} images. Starting embedding...")
print(f"Using device: {device} (GPU={device=='cuda'})\n")

for idx, image_file in enumerate(image_files): 
    try:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        embedding = image_features.cpu().numpy().flatten().tolist()
        embeddings_data[image_file] = {
            "embedding": embedding,
            "style": "Unknown"
        }
        
        print(f"[{idx+1}/{len(image_files)}] {image_file} ✓")
        
    except Exception as e:
        print(f"[{idx+1}/{len(image_files)}] {image_file} ✗ ({str(e)[:30]})")

with open("embeddings.json", "w") as f:
    json.dump(embeddings_data, f)

print(f"\n✓ Done! Saved {len(embeddings_data)} embeddings")