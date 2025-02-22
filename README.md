# Multimodal Image Retrieval System

A **production-ready** case study demonstrating how to:
1. Generate image embeddings using CLIP  
2. Index them with FAISS  
3. Build a Flask API for text-to-image retrieval  
4. Integrate a Streamlit front-end

---

## Table of Contents
1. [Repository Structure](#repository-structure)  
2. [Setup & Installation](#setup--installation)  
3. [Running the System](#running-the-system)  
   - [1) Generate Embeddings](#1-generate-embeddings)  
   - [2) Build the FAISS Index](#2-build-the-faiss-index)  
   - [3) Run the Flask API](#3-run-the-flask-api)  
   - [4) Run the Streamlit Front-End](#4-run-the-streamlit-front-end)  
4. [Running Tests](#running-tests)  
5. [Assumptions & Notes](#assumptions--notes)  
6. [Bonus: Inclusive Design](#bonus-inclusive-design)

---

## Repository Structure

- **data/images/**: Places sampled 500 images here, but they are `.gitignore`d so large files are not pushed to GitHub.
- **embeddings/**: Where `.npy`, `.faiss`, and `.txt` (image paths) are stored.
- **scripts/**: Each script focuses on one stage of the pipeline.
- **tests/**: Contains test files.
- **requirements.txt**: Lists all dependencies.


---

## Setup & Installation

1. **Clone This Repository**

   ```bash
   git clone https://github.com//leeseetja/multimodal_image_system.git
   cd multimodal_image_system
   ```

2. **Install Dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Make sure you have Python 3.8+ installed. This command installs PyTorch, FAISS, CLIP, Flask, and Streamlit.

3. **Move or Copy Images**

Place/ensure the downloaded images are in /data/images. This folder is ignored by Git, so it won’t be pushed to the repository.


## Running the System

Below are the main steps to run the multi-modal retrieval pipeline. Commands assume you’re in the root folder of this project.

1) Generate Embeddings
```bash
python ./scripts/generate_embeddings.py \
  --image_folder ./data/images \
  --sample_size 500 \
  --output_dir ./embeddings
```
* Randomly picks 500 images and uses CLIP to generate embeddings.
* Saves:
   * embeddings/image_embeddings.npy
   * embeddings/image_paths.txt

2) Build the FAISS Index
```bash
python ./scripts/build_index.py \
  --embeddings_path ./embeddings/image_embeddings.npy \
  --output_dir ./embeddings \
  --metric ip
```
* Reads the .npy file and creates image_index.faiss in the same folder.
* metric ip is effectively cosine similarity for normalised vectors.

3) Run the Flask API
```bash
python ./scripts/run_api.py
```
* The Flask server starts on http://localhost:5000.
* Keep this console open so the server remains active.

Test the API in a new terminal:
```bash
curl -X POST "http://localhost:5000/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"a futuristic cityscape","k":3}'
```
You should see a JSON response listing top-3 results with image paths and similarity scores.

4) Run the Streamlit Front-End
In another terminal:
```bash
python -m streamlit run ./scripts/run_frontend.py
```
* Streamlit runs on http://localhost:8501.
* Enter a text query, hit “Search,” and images + similarity scores should appear.

---

## Running Tests
To run tests go to /tests. You can run them with:
```bash
python -m pytest ./tests
```

---

## Assumptions & Notes

1. **CLIP Model:** Defaults to ViT-B/32. Adjust the model name if needed.
2. **FAISS Metric:** `ip` (inner product) is used with normalized vectors (cosine sim). You can switch to l2 if desired.
3. **Images:** 500 from the Kaggle test_data_v2 folder are used. Larger sets may require advanced FAISS indexes.
4. **OpenMP Conflicts:** On Windows, you may see an “OMP: Error #15” message. Setting `KMP_DUPLICATE_LIB_OK=TRUE` resolves it in some local dev setups.
5. **Local Usage:** These instructions assume everything is local. For remote demos, use a tunneling tool or cloud deployment.

---

## Bonus: Inclusive Design

To incorporate accessible features for potentially disabled users (e.g., image captioning for visually impaired users):
* **Captioning Model:** Use a Hugging Face model (like nlpconnect/vit-gpt2-image-captioning) to generate alt-text for each retrieved image.
* **Screen Reader Compatibility:** Add alt attributes or textual descriptions in the front-end so screen readers can read them aloud.
* **Speech Input:** Speech-to-text libraries can be integrated if the user can’t type queries.
These enhancements could ensure that the system caters to users with disabilities, and aligns with inclusive design principles.








