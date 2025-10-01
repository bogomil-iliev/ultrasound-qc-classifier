# Downloads and unzips the data, initial sanity check

#Installation of libraries
#PyTorch + timm for easier access to pretrained CNNs and ViTs
#Lightning fabric/pytorch-lightning to remove bilerplate in the training loop
!pip install -q torch torchvision torchaudio torchmetrics timm lightning
!pip install -q grad-cam
import torch, torchvision, random, numpy as np
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__, "| Device:", device)

# Installation of libraries for data retreival
import os, zipfile, requests, tqdm, shutil

# Download the ZIP from Mendeley
zip_url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/r6h24d2d3y-1.zip"  # direct file link
zip_path = "/content/Gallblader Diseases Dataset.zip"

if not os.path.exists(zip_path):
    with requests.get(zip_url, stream=True) as r, open(zip_path, "wb") as f:
        total = int(r.headers.get("content-length", 0))
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=1<<20), total=total//(1<<20)):
            f.write(chunk)

# Unzip the main zip to /content/data/temp
data_root = "/content/data"
temp_extract_path = "/content/data/temp" # Temporal extraction path for the first zip
if os.path.exists(data_root): shutil.rmtree(data_root)
os.makedirs(data_root, exist_ok=True) # Ensure data_root exists
os.makedirs(temp_extract_path, exist_ok=True) # Ensure temp_extract_path exists


with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(temp_extract_path)

# Find and extract the inner zip files
# The inner zips are within a subdirectory inside temp_extract_path
inner_zip_dir = None
for root, dirs, files in os.walk(temp_extract_path):
    for d in dirs:
        # Assuming the directory containing the inner zips is the only one at this level
        # or has a recognizable name pattern
        inner_zip_dir = os.path.join(root, d)
        break # Assuming we found the directory containing the zips
    if inner_zip_dir:
        break # Found the directory, exit outer loop


if inner_zip_dir and os.path.exists(inner_zip_dir):
    print(f"Found inner zip directory: {inner_zip_dir}")
    for item in os.listdir(inner_zip_dir):
        item_path = os.path.join(inner_zip_dir, item)
        if item_path.endswith(".zip"):
            print(f"Extracting inner zip: {item_path}")
            try:
                with zipfile.ZipFile(item_path, 'r') as inner_zf:
                    # Extract contents of inner zips directly into data_root
                    inner_zf.extractall(data_root)
            except zipfile.BadZipFile:
                print(f"Skipping bad zip file: {item_path}")
else:
    print(f"Could not find inner zip directory within {temp_extract_path}")


# Clean up the temporary extraction directory
if os.path.exists(temp_extract_path): shutil.rmtree(temp_extract_path)
print("Extraction complete.")

# Quick sanity-check
from collections import Counter
import glob, pandas as pd

image_paths = glob.glob(f"{data_root}/**/*.jpg", recursive=True)
labels = [p.split(os.sep)[-2] for p in image_paths]
class_counts = Counter(labels)
pd.DataFrame.from_dict(class_counts, orient='index', columns=['images']).sort_values('images')
