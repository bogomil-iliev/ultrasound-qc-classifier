# drops frames failing thresholds (blur<5 OR hf>22 OR contrast<25)
# for survivors: if contrast<30 apply CLAHE; if hf>18 apply fast NL-means
# creates train, val, test splits. 

# Setting up of QC Threshold to filter out images with unstatisfying quality.
# QC thresholds
MIN_BLUR, MAX_NOISE, MIN_CONTR = 5, 22, 25

ban_mask = (
    (df_qc.blur_var < MIN_BLUR) |
    (df_qc.hp_energy > MAX_NOISE) |
    (df_qc.contrast  < MIN_CONTR)
)
approved = df_qc[~ban_mask].path.tolist()
print(f"Approved images: {len(approved)} / {len(df_qc)}")

# Dataset class with per-image repairs (CLAHE + denoise when needed).
# Helpers
import cv2, numpy as np
from PIL import Image
def apply_clahe(arr):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(arr.astype(np.uint8))
def denoise(arr):
    return cv2.fastNlMeansDenoising(arr.astype(np.uint8), h=10)

# Custom Dataset
from torch.utils.data import Dataset
class UltrasoundDS(Dataset):
    def __init__(self, paths, tfms, qc_df):
        self.paths, self.tfms = paths, tfms
        self.qc = qc_df.set_index("path")
        self.cls2idx = {c:i for i,c in enumerate(sorted(set(labels)))}

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p   = self.paths[idx]
        arr = np.asarray(Image.open(p).convert("L"))

        # on-the-fly corrections
        row = self.qc.loc[p]
        if row.contrast < 30:
            arr = apply_clahe(arr)
        if row.hp_energy > 18:
            arr = denoise(arr)

        x = self.tfms(Image.fromarray(arr))
        y = self.cls2idx[p.split(os.sep)[-2]]
        return x, y

# Augmentation & three-way loaders (75 / 15 / 10, single-channel).
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter
import torch, numpy as np

# Transforms
IMG_SIZE = 224
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# 75 / 15 / 10 split
train_p, temp_p = train_test_split(
    approved, test_size=0.25,          # 75 % train, 25 % temp
    stratify=[p.split(os.sep)[-2] for p in approved],
    random_state=SEED
)
val_p, test_p = train_test_split(
    temp_p, test_size=0.40,            # 40 % of 25 %  => 10 % overall
    stratify=[p.split(os.sep)[-2] for p in temp_p],
    random_state=SEED
)

print(f"train {len(train_p)}  val {len(val_p)}  test {len(test_p)}")

# Datasets
train_ds = UltrasoundDS(train_p, train_tfms, df_qc)
val_ds   = UltrasoundDS(val_p,   val_tfms,   df_qc)
test_ds  = UltrasoundDS(test_p,  val_tfms,   df_qc)   # no augmentations

# Sampler & loaders
freqs = Counter([p.split(os.sep)[-2] for p in train_p])
weights = torch.tensor([1/freqs[c] for c in sorted(freqs.keys())], dtype=torch.float)
sample_w = [weights[train_ds.cls2idx[p.split(os.sep)[-2]]] for p in train_p]
sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_dl = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

