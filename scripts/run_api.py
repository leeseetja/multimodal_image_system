import os
import torch
import clip
import faiss
import numpy as np

from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths to your embeddings
EMBEDDINGS_DIR = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "image_index.faiss")
PATHS_FILE = os.path.join(EMBEDDINGS_DIR, "image_paths.txt")

# Load the FAISS index
index = faiss.read_index(INDEX_PATH)

# Load image paths
with open(PATHS_FILE, "r") as f:
    image_paths = [line.strip() for line in f.readlines()]
image_paths = np.array(image_paths)

@app.route("/")
def home():
    return "Welcome to the Multi-Modal Retrieval API!"

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query_text = data.get("query", "")
    k = data.get("k", 5)

    # Convert text to embedding
    with torch.no_grad():
        text_tokens = clip.tokenize([query_text]).to(device)
        text_embedding = model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy().astype(np.float32)

    distances, indices = index.search(text_embedding, k)
    results = []
    for i in range(k):
        idx = indices[0][i]
        results.append({
            "image_path": image_paths[idx],
            "similarity": float(distances[0][i])
        })

    return jsonify({"results": results})

@app.route("/image", methods=["GET"])
def serve_image():
    """
    Provide a path param: /image?path=some/image/path.jpg
    """
    img_path = request.args.get("path")
    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": "Image not found"}), 404
    return send_file(img_path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

