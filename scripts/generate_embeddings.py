import os
import random
import torch
import clip
from PIL import Image
import numpy as np

def generate_embeddings(image_folder, sample_size=500, model_name="ViT-B/32", output_dir="embeddings"):
    """
    1. Randomly select 'sample_size' images from 'image_folder'.
    2. Generate CLIP embeddings for those images.
    3. Save 'image_embeddings.npy' + 'image_paths.txt' in 'output_dir'.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    # Gather all images
    all_images = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    # Shuffle and select
    random.shuffle(all_images)
    selected_images = all_images[:sample_size]

    image_embeddings = []
    image_paths = []

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for filename in selected_images:
            img_path = os.path.join(image_folder, filename)

            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)

            embedding = model.encode_image(image_input)
            # Normalize for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            image_embeddings.append(embedding.cpu().numpy()[0])
            image_paths.append(img_path)

    # Convert to NumPy array
    image_embeddings = np.array(image_embeddings)
    print("Number of images processed:", len(image_paths))
    print("Embedding array shape:", image_embeddings.shape)

    # Save to disk
    emb_path = os.path.join(output_dir, "image_embeddings.npy")
    np.save(emb_path, image_embeddings)

    paths_file = os.path.join(output_dir, "image_paths.txt")
    with open(paths_file, "w") as f:
        for p in image_paths:
            f.write(p + "\n")

    print(f"Saved embeddings to: {emb_path}")
    print(f"Saved image paths to: {paths_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for images")
    parser.add_argument("--image_folder", required=True, help="Path to folder containing images")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of images to sample")
    parser.add_argument("--model_name", default="ViT-B/32", help="Which CLIP model variant to use")
    parser.add_argument("--output_dir", default="embeddings", help="Where to save embeddings & paths")
    args = parser.parse_args()

    generate_embeddings(
        image_folder=args.image_folder,
        sample_size=args.sample_size,
        model_name=args.model_name,
        output_dir=args.output_dir
    )

