# Double-Inertial Bilevel Optimization for Hybrid Deep Learning: Accelerating Chest X-Ray Classification

Official source code for the IEEE Access manuscript:

> **Kobkoon Janngam** and **Suthep Suantai**,
> *"Double-Inertial Bilevel Optimization for Hybrid Deep Learning: Accelerating Chest X-Ray Classification,"*
> **IEEE Access**, 2024. DOI: [10.1109/ACCESS.2024.0300000](https://doi.org/10.1109/ACCESS.2024.0300000)

---

## Abstract

We propose **DIPGM-VA** (Double-Inertial Proximal Gradient Method with Viscosity
Approximation), a novel convex bilevel optimization algorithm that combines
double-inertial extrapolation with viscosity approximation and achieves strong
convergence in real Hilbert spaces. We integrate DIPGM-VA with a hybrid deep
learning architecture — an **EfficientNetV2-B0** feature extractor followed by
an **Extreme Learning Machine (ELM)** classifier — for binary pneumonia
classification on pediatric chest X-rays. On the Kermany *et al.* (2018)
dataset, the proposed method achieves **F1 = 0.9001** with approximately
**10× speed-up** in classifier training time compared with an end-to-end SGD
baseline, while also outperforming the iBiG-SAM and aiBiG-SAM bilevel
baselines on every metric.

## Key contributions

1. **DIPGM-VA** — a new convex bilevel optimization algorithm with strong
   convergence via double-inertial extrapolation on the post-proximal
   sequence + viscosity approximation.
2. **Hybrid CNN + bilevel ELM** architecture that decouples representation
   learning from classifier optimization, enabling an order-of-magnitude
   speed-up with no loss in predictive quality.
3. A fair, reproducible comparison of DIPGM-VA against iBiG-SAM and aiBiG-SAM
   under identical backbone features, hyperparameters, and seeds.

---

## Repository layout

```
CNN-ELM-DIPGM-VA/
├── README.md                    - this file
├── LICENSE                      - MIT license
├── .gitignore
├── requirements.txt             - pip dependencies
├── environment.yml              - conda environment
├── CITATION.cff                 - machine-readable citation metadata
├── notebooks/
│   └── CNN_ELM_DIPGM_VA.ipynb   - end-to-end reproducible notebook
├── scripts/
│   └── build_clean_notebook.py  - regenerates the clean notebook
├── docs/
│   ├── CHANGELOG.md             - refactor history vs. the original code
│   └── code_availability.tex    - paragraph to paste into the manuscript
└── figures/                     - selected figures from the paper
```

---

## Prerequisites

- **Python** 3.10+ (tested on 3.12)
- **PyTorch** 2.4+ (tested on 2.9.0 + CUDA 12.6)
- An NVIDIA GPU with at least **8 GB VRAM** is recommended but not required
  (the notebook auto-detects and falls back to CPU).
- A **Kaggle API token** for `kagglehub` to download the dataset. Place it at:
  - Linux / macOS: `~/.kaggle/kaggle.json`
  - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
  (Kaggle → Account → Create New API Token)

## Installation

### Option A — pip
```bash
git clone https://github.com/KobkoonCoding/CNN-ELM-DIPGM-VA.git
cd CNN-ELM-DIPGM-VA
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate

# Install PyTorch matching your CUDA version from https://pytorch.org, then:
pip install -r requirements.txt
```

### Option B — conda
```bash
git clone https://github.com/KobkoonCoding/CNN-ELM-DIPGM-VA.git
cd CNN-ELM-DIPGM-VA
conda env create -f environment.yml
conda activate dipgm-va
```

## Dataset preparation

The notebook downloads the Kermany *et al.* (2018) chest X-ray dataset
automatically via `kagglehub` on first execution:

- **Source:** <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>
- **Size:** 5,856 anterior-posterior chest radiographs
- **Population:** pediatric patients aged 1–5 (Guangzhou Women and Children's
  Medical Center)
- **Classes:** `NORMAL` vs. `PNEUMONIA`
- **Split used in the paper:** original `train/` is re-split 85 % / 15 %
  (stratified, seed 42) into training and validation; the original `test/`
  (624 images) is **locked** and used only for final evaluation.

No manual download is required, but you must be signed in with a valid Kaggle
API token.

## How to run

1. Launch Jupyter:
   ```bash
   jupyter notebook notebooks/CNN_ELM_DIPGM_VA.ipynb
   ```
2. **First run:** execute every cell top-to-bottom. Cells 11–14 perform the
   two-stage fine-tuning of EfficientNetV2-B0 and cache the artifacts under
   `artifacts/`:
   - `artifacts/best_backbone.pth`
   - `artifacts/feature_exports/features_v6.pt`
   - `artifacts/feature_exports/features_v6_scaled.npz`
   - `artifacts/training_history.csv`
   - `artifacts/timing_first_run.json`
3. **Subsequent runs:** skip Cells 11–14 and execute **Cell 15 (LOAD)**
   instead, then continue from Cell 16. This reuses the cached backbone and
   features and takes only seconds.

### Expected reproduction time (single NVIDIA Tesla P100, 16 GB VRAM)

| Stage | Time |
|-------|------|
| Backbone fine-tuning (Stage 1 + Stage 2) | ~1,800 s |
| Feature extraction (1,280-D)             | ~180 s   |
| DIPGM-VA bilevel solver (500 iters)      | **< 1 s** |
| Full notebook (first run)                | ~35 min  |
| Full notebook (LOAD mode, subsequent)    | ~1 min   |

## Results

Test-set metrics (n = 624 images, locked split):

| Model                                        | Accuracy | Precision | Recall     | F1         |
|----------------------------------------------|:--------:|:---------:|:----------:|:----------:|
| EfficientNetV2-B0 + Standard head (SGD)      | 0.8590   | **0.9242**| 0.8436     | 0.8820     |
| EfficientNetV2-B0 + ELM (**DIPGM-VA**)       |**0.8638**| 0.8308    | **0.9821** | **0.9001** |
| EfficientNetV2-B0 + ELM (iBiG-SAM)           | 0.8574   | 0.8308    | 0.9692     | 0.8947     |
| EfficientNetV2-B0 + ELM (aiBiG-SAM)          | 0.8574   | 0.8308    | 0.9692     | 0.8947     |

DIPGM-VA achieves the highest accuracy, recall, and F1-score, while delivering
approximately a **10×** speed-up in classifier training over the SGD baseline
(sub-second solve vs. ~10 s SGD epoch after backbone features are cached).

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Janngam2024DIPGMVA,
  author  = {Janngam, Kobkoon and Suantai, Suthep},
  title   = {Double-Inertial Bilevel Optimization for Hybrid Deep Learning:
             Accelerating Chest X-Ray Classification},
  journal = {IEEE Access},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.0300000},
  url     = {https://doi.org/10.1109/ACCESS.2024.0300000}
}
```

## Code availability

The source code accompanying this manuscript is openly available on GitHub at
`https://github.com/KobkoonCoding/CNN-ELM-DIPGM-VA` and permanently archived
on Zenodo at `https://doi.org/10.5281/zenodo.XXXXXXX` (DOI minted from the
corresponding GitHub release tag). The Kermany *et al.* (2018) chest X-ray
dataset used in this study is publicly available at
<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>.

## License

This project is released under the **MIT License** — see
[`LICENSE`](./LICENSE) for the full text. The MIT license applies to the code
only; the manuscript and its figures remain under IEEE copyright.

## Acknowledgements

- Kermany *et al.* (2018) for the open chest X-ray dataset.
- The `timm` and `PyTorch` teams for the EfficientNetV2-B0 implementation.
- Department of Mathematics, Faculty of Science, Chiang Mai University.
- Research Center in Optimization and Computational Intelligence for Big Data
  Prediction, Chiang Mai University.

## Contact

For questions about the code or paper, please contact the corresponding
author:

> **Dr. Kobkoon Janngam** — `kobkoon.j@gmail.com`
> Department of Mathematics, Faculty of Science, Chiang Mai University, Thailand
