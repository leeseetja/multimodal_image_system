import os
import pytest
import numpy as np
import faiss
from scripts.build_index import build_faiss_index

TEST_EMBEDDINGS = "tests/test_data/embeddings/image_embeddings.npy"
TEST_OUTPUT = "tests/test_data/embeddings"

def test_build_faiss_index():
    if not os.path.isfile(TEST_EMBEDDINGS):
        pytest.skip("Embeddings file not found, skipping index test.")

    build_faiss_index(
        embeddings_path=TEST_EMBEDDINGS,
        output_dir=TEST_OUTPUT,
        metric="ip"
    )

    index_path = os.path.join(TEST_OUTPUT, "image_index.faiss")
    assert os.path.isfile(index_path), "FAISS index file not created."

    # Optionally load the index with faiss to ensure it's valid
    index = faiss.read_index(index_path)
    assert index.ntotal > 0, "Index should contain vectors."

