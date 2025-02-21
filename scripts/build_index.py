# scripts/build_index.py

import os
import faiss
import numpy as np

def build_faiss_index(embeddings_path, output_dir="embeddings", metric="ip"):
    """
    1. Loads embeddings from .npy file.
    2. Builds a FAISS index (inner product or L2).
    3. Saves 'image_index.faiss' in 'output_dir'.
    """
    image_embeddings = np.load(embeddings_path)
    dim = image_embeddings.shape[1]

    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError("Unsupported metric. Use 'ip' or 'l2'.")

    index.add(image_embeddings)
    print("Total vectors in the index:", index.ntotal)

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "image_index.faiss")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build a FAISS index from embeddings.")
    parser.add_argument("--embeddings_path", required=True, help="Path to the .npy embeddings.")
    parser.add_argument("--output_dir", default="embeddings", help="Where to save FAISS index.")
    parser.add_argument("--metric", default="ip", help="FAISS metric: ip or l2")
    args = parser.parse_args()

    build_faiss_index(args.embeddings_path, args.output_dir, args.metric)

