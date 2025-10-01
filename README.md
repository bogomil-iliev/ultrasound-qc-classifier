# Ultrasound QC Classifier — data-centric pipeline with EfficientNet (PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/bogomil-iliev/ultrasound-qc-classifier/blob/main/notebooks/ain7301_ultrasound_qc_pipeline.ipynb
)
![Python](https://img.shields.io/badge/python-3.10+-informational)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Quality-controlled classification of **gallbladder ultrasound** (9 classes) with:
- **Blur/Noise/Contrast QC** (drop blurred/noisy/low-contrast frames; selective **CLAHE** / **NL-means** when useful)
- **EfficientNet-B0** baseline vs **tf-EfficientNet-V2-S**
- **Macro-recall** as the primary metric for imbalance

> Based on my MSc Applied AI assignment (AIN7301). Full report in `docs/ain7301_report.pdf`.

## Highlights
- QC thresholds: blur var < **5**, HF energy > **22%**, contrast < **25** then drop; **CLAHE** if contrast < **30**; **NL-means** if HF > **18**.
- 75/15/10 split (image-level), class-balanced sampling, **early stopping (patience=3)**.
- Two-stage fine-tuning: freeze head then full fine-tune (AdamW, mixed precision).

## Quickstart

**Colab (recommended):** click the badge above and run:
1) Download dataset > 2) EDA & QC > 3) Split 75/15/10 > 4) Train (B0, V2-S) > 5) Evaluate.

**Local (minimal):**
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/qc_analyse.py          # computes blur/HF/contrast stats
python scripts/cache_preprocess.py    # selective CLAHE/NL-means, resize 224
python scripts/make_splits.py         # writes split CSVs or lists
python scripts/train.py --model tf_efficientnetv2_s
python scripts/evaluate.py
```
## Data

This repo **does not include images**. See [data/README.md](data/README.md).

## Results

<img width="497" height="61" alt="image" src="https://github.com/user-attachments/assets/d2927cbc-fd65-4451-a983-94c264a97277" />

**Baseline Model Training Curves**

<img width="544" height="762" alt="image" src="https://github.com/user-attachments/assets/cb3b3ad7-f0cb-4b82-a43e-c2a9e5f85070" />

**tf_efficientnetv2_s Training Curves**

<img width="538" height="762" alt="image" src="https://github.com/user-attachments/assets/2ca41fdf-95e8-40cf-bbf1-a49568816d4b" />

**Confusion Matrix Test**

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/19130527-3e4b-4183-afde-92e74caf2aac" />

**Per-class Recall Test**

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/a1fba896-dea9-4967-8404-a8c24cb00125" />

**Prediction on Random Pathology Images from Google Images Test**

<img width="331" height="614" alt="image" src="https://github.com/user-attachments/assets/a97ca0ae-df8c-4cb5-9d3c-685a816cb736" />

<img width="357" height="310" alt="image" src="https://github.com/user-attachments/assets/b2280480-0e66-4f4f-aee2-e280c315de0d" />

## Repo map
```pqsql
notebooks/ … end-to-end pipeline 
scripts/   … download, QC analysis (EDA), pre-processing, splits, train, evaluate, predict demo
configs/   … default.yaml (QC thresholds, training knobs)
docs/      … report + figures, test files
```
## Ethics and notes

Research/education only — not a medical device. Image-level splits (no patient IDs available in this dataset). QC aims to reject unreadable frames and apply minimal, principled fixes.

## License
**MIT**

---


### Scripts
**`scripts/download_data.py` (download and initial sanity check)**
```python
# downloads the dataset
# unzips it
# initial sanity check
```

**`scripts/qc_analyse.py` (basic EDA, computes blur/HF/contrast)**
```python
# Basic EDA
# reads images under data/images/**.jpg, computes:
# - resolution distributions
# - grayscale vs. color analysis
# - Variance of Laplacian (blur)
# - High-frequency energy (FFT proportion)
# - RMS contrast
```
<img width="268" height="185" alt="image" src="https://github.com/user-attachments/assets/46b12c25-eeee-469a-b38d-b8bfdcb3c15a" />

<img width="1125" height="590" alt="image" src="https://github.com/user-attachments/assets/b4cea03a-2d09-4270-86e6-c1a5e62271e0" />

<img width="588" height="490" alt="image" src="https://github.com/user-attachments/assets/a195c6de-6b61-43ca-ae88-9f95e061b90b" />

<img width="704" height="547" alt="image" src="https://github.com/user-attachments/assets/f8602d8e-8862-4511-8232-2b0ec73eafe2" />

<img width="851" height="153" alt="image" src="https://github.com/user-attachments/assets/688d2a9e-0604-4a53-bcce-48e81d4abad9" />

<img width="1590" height="390" alt="image" src="https://github.com/user-attachments/assets/d49d1574-77c7-40ea-8843-bbb6478e0505" />

<img width="872" height="287" alt="image" src="https://github.com/user-attachments/assets/d5dd89f0-9ce0-4731-b6ed-cd4b00374d4e" />





**scripts/preprocess_splits.py (selective CLAHE / NL-means, resize 224)**
```python
# drops frames failing thresholds (blur<5 OR hf>22 OR contrast<25)
# for survivors: if contrast<30 apply CLAHE; if hf>18 apply fast NL-means
# creates train, val, test splits. 
```

**scripts/train.py**
```python
# trains EfficientNet-B0 and/or tf_efficientnetv2_s
# two-stage: freeze head (2 epochs) -> full fine-tune (30 epochs, patience=3)
# class-weighted CE, mixed precision, AdamW; logs macro-recall (torchmetrics)
# saves best checkpoint and curves
```

**scripts/evaluate.py**
```python
# loads best checkpoint, computes macro-recall on test
# plots confusion matrix & per-class recall to docs/figures
```

**scripts/predict_demo.py**
```python
# loads checkpoint; runs inference for a few example JPGs; prints top-1 + confidence
```
### Citation
[➡️ Cite this repository](./CITATION.cff)



