import os
import pytest
import shutil
from PIL import Image
import numpy as np
from scripts.generate_embeddings import generate_embeddings

TEST_IMAGE_FOLDER = "tests/test_data/images"
TEST_EMBEDDINGS_FOLDER = "tests/test_data/embeddings"

@pytest.fixture(scope="session", autouse=True)
def setup_test_images():
    """
    Creates a small test_images folder with 2-3 dummy images for embedding.
    Cleans up after tests.
    """
    os.makedirs(TEST_IMAGE_FOLDER, exist_ok=True)
    # Create small dummy images
    img1 = Image.new("RGB", (64, 64), color="red")
    img1.save(os.path.join(TEST_IMAGE_FOLDER, "test1.jpg"))
    img2 = Image.new("RGB", (64, 64), color="blue")
    img2.save(os.path.join(TEST_IMAGE_FOLDER, "test2.jpg"))

    yield  # run tests

    # Cleanup
    shutil.rmtree("tests/test_data", ignore_errors=True)

def test_generate_embeddings():
    # Run the function directly
    generate_embeddings(
        image_folder=TEST_IMAGE_FOLDER,
        sample_size=2,
        model_name="ViT-B/32",
        output_dir=TEST_EMBEDDINGS_FOLDER
    )

    # Check if output files exist
    emb_path = os.path.join(TEST_EMBEDDINGS_FOLDER, "image_embeddings.npy")
    paths_txt = os.path.join(TEST_EMBEDDINGS_FOLDER, "image_paths.txt")

    assert os.path.isfile(emb_path), "Embeddings file not found."
    assert os.path.isfile(paths_txt), "Image paths file not found."

    arr = np.load(emb_path)
    assert arr.shape[0] == 2, "Expected embeddings for 2 images."

