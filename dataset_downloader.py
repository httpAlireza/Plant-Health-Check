import os
import zipfile
import kaggle

# Ensure Kaggle API credentials are set
# Place kaggle.json in ~/.kaggle/ or set environment variables
# The kaggle.json file (API token) can be downloaded from: https://www.kaggle.com/account

DATASET_ROOT_DIR = './dataset'

datasets = {
    "plantvillage-dataset": "abdallahalidev/plantvillage-dataset",
    "leaf-detection": "alexo98/leaf-detection"
}

os.makedirs(DATASET_ROOT_DIR, exist_ok=True)

for folder_name, kaggle_id in datasets.items():
    print(f"\n📥 Downloading dataset: {kaggle_id}")

    kaggle.api.dataset_download_files(kaggle_id, path=DATASET_ROOT_DIR, unzip=False)

    downloaded_zip = os.path.join(DATASET_ROOT_DIR, kaggle_id.split("/")[-1] + ".zip")
    if not os.path.exists(downloaded_zip):
        raise FileNotFoundError(f"❌ Could not find downloaded zip: {downloaded_zip}")

    extract_path = os.path.join(DATASET_ROOT_DIR, folder_name)
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(downloaded_zip)

    print(f"✅ Dataset '{folder_name}' extracted to '{extract_path}' and zip removed.")
