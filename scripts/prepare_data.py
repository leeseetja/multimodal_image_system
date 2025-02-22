"""
prepare_data.py

Description:
    This script automates downloading the required dataset from kaggle,
    storing it locally, and returning the path to the folder containing images.

Notes:
    - Adjust the Kaggle dataset ID and subfolder path as necessary.
    - Requires 'kagglehub' or the Kaggle API, depending on your approach.
"""

import kagglehub
import os

def download_data(dataset_id="alessandrasala79/ai-vs-human-generated-dataset", subfolder="test_data_v2"):
    """
    Uses kagglehub to download the specified dataset and returns
    the path to the chosen subfolder test_data_v2.
    """
    path = kagglehub.dataset_download(dataset_id)
    data_path = os.path.join(path, subfolder)
    print("Data subfolder path:", data_path)
    return data_path

if __name__ == "__main__":
    data_path = download_data()
    print(f"Data downloaded to {data_path}")

