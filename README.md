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

multimodal_image_system/
├── data/
│   └── images/         		# Downloaded images (git-ignored)
├── embeddings/
│   ├── image_embeddings.npy		# Generated embeddings
│   ├── image_index.faiss		# FAISS index
│   └── image_paths.txt			# Image file paths
├── scripts/
│   ├── prepare_data.py        		# Downloads required Kaggle data
│   ├── generate_embeddings.py		# Generates CLIP embeddings
│   ├── build_index.py			# Builds the FAISS index
│   ├── run_api.py            		# Flask retrieval API
│   └── run_frontend.py       		# Streamlit front-end
├── tests/
│   └── test_api.py 			# tests          
├── requirements.txt
└── README.md


- **data/images/**: Places sampled 500 images here, but they are `.gitignore`d so large files are not pushed to GitHub.
- **embeddings/**: Where `.npy`, `.faiss`, and `.txt` (image paths) are stored.
- **scripts/**: Each script focuses on one stage of the pipeline.
- **tests/**: Contains test files.
- **requirements.txt**: Lists all dependencies.


---

## Setup & Installation

1. **Clone This Repository**

   ```bash
   git clone https://github.com/YourUsername/multimodal_image_system.git
   cd multimodal_image_system


2. **Install Dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt















