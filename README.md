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




