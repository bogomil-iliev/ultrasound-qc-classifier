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

