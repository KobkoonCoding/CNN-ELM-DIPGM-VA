"""
Build Code_clean.ipynb from Code.ipynb:
- Translates all Thai markdown/comments to English
- Fixes hardcoded Kaggle paths in LOAD cell
- Removes dead LASSO config block
- Aligns terminology with the IEEE Access manuscript (DIPGM-VA, EfficientNetV2-B0, Kermany)
"""
import json
from pathlib import Path

SRC = Path("Code.ipynb")
DST = Path("Code_clean.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

# ──────────────────────────────────────────────────────────────────
# 0. Title
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""# EfficientNetV2-B0 + ELM via Convex Bilevel Optimization (DIPGM-VA)

**Paper:** *Double-Inertial Bilevel Optimization for Hybrid Deep Learning: Accelerating Chest X-Ray Classification* — Janngam & Suantai, IEEE Access.

**Proposed method:** EfficientNetV2-B0 backbone + Extreme Learning Machine (ELM) classifier whose output weights are solved by **DIPGM-VA** (Double-Inertial Proximal Gradient Method with Viscosity Approximation).

## Models Compared
| # | Model | Classifier solver |
|---|-------|-------------------|
| 1 | EfficientNetV2-B0 + Standard head | SGD (iterative, end-to-end) |
| 2 | EfficientNetV2-B0 + ELM (**DIPGM-VA**) | Double-inertial proximal + viscosity (**proposed**) |
| 3 | EfficientNetV2-B0 + ELM (iBiG-SAM) | Single-inertial bilevel |
| 4 | EfficientNetV2-B0 + ELM (aiBiG-SAM) | Alternating-inertial bilevel |

**Dataset:** Kermany *et al.* (2018) chest-xray-pneumonia (5,856 pediatric chest radiographs).

### Run guide
| Run | Instructions |
|-----|--------------|
| First  | Execute every cell in order. Cells 11–14 train the backbone and cache artifacts (`best_backbone.pth`, `features_v6_scaled.npz`). |
| Second+ | Skip the training cells (11–14) and execute the **LOAD** cell (Cell 15) instead, then continue from Cell 16. |
"""))

# ──────────────────────────────────────────────────────────────────
# 1 — Install dependencies
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 1 — Install dependencies

Install the packages required by the full pipeline:
- `kagglehub` — downloads the Kermany dataset from Kaggle
- `timm` — pretrained EfficientNetV2-B0 backbone
- `opencv-python-headless` — CLAHE preprocessing
- `scikit-learn`, `seaborn` — metrics and plotting
"""))

cells.append(code(
"""# Uncomment the line below the first time you run the notebook in a fresh environment.
# !pip -q install kagglehub timm scikit-learn seaborn opencv-python-headless
"""))

# ──────────────────────────────────────────────────────────────────
# 2 — Imports
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 2 — Import libraries

All libraries used throughout the pipeline:
- **PyTorch** — backbone training and ELM tensor algebra on GPU
- **timm** — pretrained EfficientNetV2-B0
- **scikit-learn** — metrics, `StandardScaler`, `StratifiedShuffleSplit`
- **OpenCV** — CLAHE preprocessing
- **kagglehub** — dataset download
"""))

cells.append(code(
"""import os, sys, time, copy, json, warnings, math, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.datasets as datasets

import timm
from timm.data import resolve_model_data_config

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, balanced_accuracy_score)

import kagglehub

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__} | Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
"""))

# ──────────────────────────────────────────────────────────────────
# 3 — Configuration
# ──────────────────────────────────────────────────────────────────
cells.append(md(
r"""## Cell 3 — Configuration & hyperparameters

All pipeline hyperparameters live in this single cell so readers can audit and modify them in one place.

### Data-split strategy
- Original `train/` (5,216 images) → stratified **85 % train / 15 % val** split via `StratifiedShuffleSplit` (4,433 train + 783 val).
- Original `val/` (16 images) → **discarded** (too small for reliable tuning).
- Original `test/` (624 images) → **locked**, used only for final evaluation; never touched during training or model selection.

### Backbone training (two-stage fine-tuning)
- **Stage 1:** freeze the backbone, train only the classifier head so the head learns an ImageNet → {Normal, Pneumonia} mapping without disturbing pretrained features.
- **Stage 2:** unfreeze the last 50 % of parameter tensors and fine-tune at a small learning rate so higher-level features adapt to the chest X-ray domain while low-level features (edges, textures) remain fixed.

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Improves local contrast in X-rays so infiltrates become more visible.
- `clip_limit=2.0` bounds histogram amplification (prevents noise blow-up).
- `tile_grid_size=(8, 8)` divides each image into 64 regions, each equalised independently.

### Bilevel ELM hyperparameter grid
- Hidden sizes $L \in \{512, 1024, 2048\}$
- Regularisation $\lambda \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$
- Activation: ReLU
- Maximum iterations: 500

### Artifact locations
All artifacts are written under `./artifacts/` so the notebook is fully portable.
"""))

cells.append(code(
r"""# ══════════════════════════════════════════════════════════════════
#  Dataset & model identifiers
# ══════════════════════════════════════════════════════════════════
DATASET_HANDLE = "paultimothymooney/chest-xray-pneumonia"
MODEL_NAME     = "tf_efficientnetv2_b0.in1k"
NUM_CLASSES    = 2
SEED           = 42

# ══════════════════════════════════════════════════════════════════
#  Data split & loader settings
# ══════════════════════════════════════════════════════════════════
VAL_SPLIT    = 0.15
BATCH_SIZE   = 32
NUM_WORKERS  = 2

# ══════════════════════════════════════════════════════════════════
#  CLAHE preprocessing
# ══════════════════════════════════════════════════════════════════
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE  = (8, 8)

# ══════════════════════════════════════════════════════════════════
#  Stage 1 — head only (backbone frozen)
# ══════════════════════════════════════════════════════════════════
STAGE1_EPOCHS       = 10
STAGE1_LR           = 1e-3
STAGE1_LABEL_SMOOTH = 0.10
STAGE1_PATIENCE     = 4

# ══════════════════════════════════════════════════════════════════
#  Stage 2 — partial fine-tuning (last 50% unfrozen)
# ══════════════════════════════════════════════════════════════════
STAGE2_EPOCHS            = 15
STAGE2_LR                = 2e-5
STAGE2_FINETUNE_FRACTION = 0.50
STAGE2_LABEL_SMOOTH      = 0.05
STAGE2_PATIENCE          = 5
GRAD_CLIP_NORM           = 1.0
WEIGHT_DECAY             = 1e-4

# ══════════════════════════════════════════════════════════════════
#  Data augmentation (training only)
# ══════════════════════════════════════════════════════════════════
AUG_HFLIP_P      = 0.50
AUG_ROTATION_DEG = 10
AUG_TRANSLATE    = 0.05
AUG_BRIGHTNESS   = 0.20
AUG_CONTRAST     = 0.20

# ══════════════════════════════════════════════════════════════════
#  Bilevel ELM grid (DIPGM-VA / iBiG-SAM / aiBiG-SAM share the best config)
# ══════════════════════════════════════════════════════════════════
ELM_CLASS_WEIGHT     = "balanced"
BILEVEL_HIDDEN_GRID  = [512, 1024, 2048]
BILEVEL_REG_GRID     = [1e-5, 1e-4, 1e-3, 1e-2]
BILEVEL_ACT_GRID     = ["relu"]
BILEVEL_MAX_ITER     = 500

# ══════════════════════════════════════════════════════════════════
#  Artifact paths (all under ./artifacts)
# ══════════════════════════════════════════════════════════════════
ARTIFACTS_DIR   = Path("artifacts")
FEATURES_DIR    = ARTIFACTS_DIR / "feature_exports"
FEATURES_FILE   = FEATURES_DIR  / "features_v6.pt"
FEATURES_SCALED = FEATURES_DIR  / "features_v6_scaled.npz"
SPLIT_CSV       = FEATURES_DIR  / "split_index.csv"
BACKBONE_PATH   = ARTIFACTS_DIR / "best_backbone.pth"
TIMING_JSON     = ARTIFACTS_DIR / "timing_first_run.json"
HISTORY_CSV     = ARTIFACTS_DIR / "training_history.csv"
FIG_DIR         = Path("paper_figures")
TABLE_DIR       = Path("paper_tables")
for d in (ARTIFACTS_DIR, FEATURES_DIR, FIG_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  Reproducibility — deterministic seeds
# ══════════════════════════════════════════════════════════════════
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 64)
print("  CONFIG — EfficientNetV2-B0 + Bilevel ELM (DIPGM-VA)")
print("=" * 64)
print(f"  Model        : {MODEL_NAME}")
print(f"  Seed         : {SEED}")
print(f"  CLAHE        : clip={CLAHE_CLIP_LIMIT}, tile={CLAHE_TILE_SIZE}")
print(f"  Stage 1      : {STAGE1_EPOCHS} ep  LR={STAGE1_LR:.0e}")
print(f"  Stage 2      : {STAGE2_EPOCHS} ep  LR={STAGE2_LR:.0e}  unfreeze={STAGE2_FINETUNE_FRACTION*100:.0f}%")
print(f"  Bilevel ELM  : max_iter={BILEVEL_MAX_ITER}, grid={len(BILEVEL_HIDDEN_GRID)*len(BILEVEL_REG_GRID)} combos")
print(f"  Artifacts    : {ARTIFACTS_DIR.resolve()}")
print("=" * 64)
"""))

# ──────────────────────────────────────────────────────────────────
# 4 — Download
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 4 — Download the Kermany chest X-ray dataset

Downloads the Kermany *et al.* (2018) pediatric chest X-ray dataset from Kaggle via `kagglehub`. Requires a valid Kaggle API token — see the repository README for setup instructions.

Folder layout after download:
```
chest_xray/
├── train/     (5,216 images — NORMAL=1,341, PNEUMONIA=3,875)
├── val/       (16 images   — discarded, too small)
└── test/      (624 images  — NORMAL=234, PNEUMONIA=390)
```
"""))

cells.append(code(
"""data_root = Path(kagglehub.dataset_download(DATASET_HANDLE)) / "chest_xray"
if not data_root.exists():
    data_root = Path(kagglehub.dataset_download(DATASET_HANDLE))
    for p in data_root.rglob("chest_xray"):
        if p.is_dir() and (p / "train").exists():
            data_root = p
            break

print(f"Data root: {data_root}")
for split in ["train", "val", "test"]:
    p = data_root / split
    if p.exists():
        n = (sum(1 for _ in p.rglob("*.jpeg"))
             + sum(1 for _ in p.rglob("*.jpg"))
             + sum(1 for _ in p.rglob("*.png")))
        print(f"  {split:6s}: {n} images")
"""))

# ──────────────────────────────────────────────────────────────────
# 5 — Stats & class weights
# ──────────────────────────────────────────────────────────────────
cells.append(md(
r"""## Cell 5 — Dataset statistics & class weights

Counts the images per class per split and computes **inverse-frequency class weights**

$$w_c = \frac{N_{\text{total}}}{C \cdot N_c},$$

with $C = 2$ classes (Normal, Pneumonia). These weights enter `CrossEntropyLoss(weight=...)` during backbone training to compensate for the ~74 % / 26 % pneumonia / normal class imbalance.
"""))

cells.append(code(
"""VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(root):
    rows = []
    for split in ["train", "val", "test"]:
        for cls_dir in sorted((root / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            n = sum(1 for f in cls_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in VALID_EXTS)
            rows.append({"split": split, "class": cls_dir.name, "n": n})
    return pd.DataFrame(rows)

count_df       = count_images(data_root)
pivot          = count_df.pivot(index="split", columns="class", values="n").fillna(0).astype(int)
pivot["total"] = pivot.sum(axis=1)
CLASS_NAMES    = sorted(c for c in pivot.columns if c != "total")
N_TRAIN_ORIG   = int(pivot.loc["train", "total"])
N_TEST         = int(pivot.loc["test",  "total"])

print("Dataset statistics:")
try:
    display(pivot)
except NameError:
    print(pivot)

train_counts     = {cls: int(pivot.loc["train", cls]) for cls in CLASS_NAMES}
class_weights_np = np.array([
    N_TRAIN_ORIG / (len(CLASS_NAMES) * train_counts[c]) for c in CLASS_NAMES
])
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)

print(f"\\nClass weights: {dict(zip(CLASS_NAMES, class_weights_np.round(4)))}")
"""))

# ──────────────────────────────────────────────────────────────────
# Extra imports / plot style (formerly cell 11)
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 5b — Figure style & additional metrics imports

Configures journal-quality plot defaults (serif fonts, 300 dpi on save) and a colour-blind friendly (Okabe–Ito) palette used for all figures in the paper.
"""))

cells.append(code(
"""from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss,
)

# Journal-quality matplotlib defaults
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Okabe-Ito colour-blind-friendly palette
COLORS  = {"Std Head": "#000000", "DIPGM-VA": "#D55E00",
           "iBiG-SAM": "#0072B2", "aiBiG-SAM": "#CC79A7"}
MARKERS = {"Std Head": "D", "DIPGM-VA": "o", "iBiG-SAM": "s", "aiBiG-SAM": "^"}

print(f"Plot style ready. Figures -> {FIG_DIR}/  Tables -> {TABLE_DIR}/")
"""))

# ──────────────────────────────────────────────────────────────────
# 6 — CLAHE
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 6 — CLAHE transform & preprocessing pipeline

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE enhances local contrast in X-rays by:
1. Partitioning the image into tiles (8 × 8 = 64 regions).
2. Computing the histogram of each tile independently.
3. Clipping each histogram at `clip_limit` to bound noise amplification.
4. Redistributing the clipped excess uniformly.
5. Bilinearly interpolating between tiles to remove block artifacts.

For RGB images the image is converted to LAB colour space and CLAHE is applied to the **L channel only** so the chromatic components are untouched.

### Transform pipeline
```
Train: CLAHE -> Resize(208) -> RandomCrop(192) -> HFlip -> Rotate -> Affine -> ColorJitter -> ToTensor -> Normalize
Eval : CLAHE -> Resize(192)                                                 -> ToTensor -> Normalize
```
"""))

cells.append(code(
'''class CLAHETransform:
    """Apply CLAHE on the L channel of LAB (for RGB) or directly (for grayscale)."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                     tileGridSize=self.tile_grid_size)
            img_np = clahe.apply(img_np)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                     tileGridSize=self.tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_np)

    def __repr__(self):
        return f"CLAHETransform(clip={self.clip_limit}, tile={self.tile_grid_size})"

# Read the backbone's expected input size and normalisation stats from timm
_dummy     = timm.create_model(MODEL_NAME, pretrained=False)
_cfg       = resolve_model_data_config(_dummy)
del _dummy
INPUT_SIZE = _cfg["input_size"][1]
NORM_MEAN  = _cfg["mean"]
NORM_STD   = _cfg["std"]

clahe_tf = CLAHETransform(CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE)

train_transform = T.Compose([
    clahe_tf,
    T.Resize((INPUT_SIZE + 16, INPUT_SIZE + 16)),
    T.RandomCrop(INPUT_SIZE),
    T.RandomHorizontalFlip(p=AUG_HFLIP_P),
    T.RandomRotation(degrees=AUG_ROTATION_DEG),
    T.RandomAffine(degrees=0, translate=(AUG_TRANSLATE, AUG_TRANSLATE)),
    T.ColorJitter(brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST,
                  saturation=0.0, hue=0.0),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])
eval_transform = T.Compose([
    clahe_tf,
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

print(f"EfficientNetV2-B0: input={INPUT_SIZE}x{INPUT_SIZE}")
print(f"CLAHE: clip_limit={CLAHE_CLIP_LIMIT}, tile={CLAHE_TILE_SIZE}")

# ══════════════════════════════════════════════════════════════════
#  Figure 2 — CLAHE preprocessing effect (original / enhanced / histogram)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(CLASS_NAMES), 3,
                         figsize=(11, 3.5 * len(CLASS_NAMES)))
if len(CLASS_NAMES) == 1:
    axes = axes.reshape(1, -1)

for i, cls in enumerate(CLASS_NAMES):
    cls_dir = data_root / "train" / cls
    imgs = sorted([f for f in cls_dir.rglob("*")
                   if f.is_file() and f.suffix.lower() in VALID_EXTS])
    img_path = imgs[len(imgs) // 3]  # pick a representative image (1/3 of the way through)
    img = np.array(Image.open(img_path).convert("RGB"))

    # Apply CLAHE on the L channel of LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    axes[i, 0].imshow(img);      axes[i, 0].set_title(f"{cls} - Original");   axes[i, 0].axis("off")
    axes[i, 1].imshow(enhanced); axes[i, 1].set_title(f"{cls} - After CLAHE"); axes[i, 1].axis("off")

    gray_orig = cv2.cvtColor(img,      cv2.COLOR_RGB2GRAY)
    gray_enh  = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    axes[i, 2].hist(gray_orig.ravel(), bins=128, alpha=0.5,
                    label="Original", color="#0072B2", density=True)
    axes[i, 2].hist(gray_enh.ravel(),  bins=128, alpha=0.5,
                    label="CLAHE",    color="#D55E00", density=True)
    axes[i, 2].set_title(f"{cls} - Histogram")
    axes[i, 2].legend(fontsize=9)
    axes[i, 2].set_xlabel("Pixel Intensity")
    axes[i, 2].set_ylabel("Density")

plt.suptitle(f"Fig. 2: Effect of CLAHE preprocessing (clip={CLAHE_CLIP_LIMIT}, tile={CLAHE_TILE_SIZE})",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_clahe_effect.png")
plt.savefig(FIG_DIR / "fig2_clahe_effect.pdf")
plt.show()
print(f"Saved: {FIG_DIR}/fig2_clahe_effect.png / .pdf")
'''))

# ──────────────────────────────────────────────────────────────────
# 7 — Split & loaders
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 7 — Stratified split & DataLoaders

### Split strategy
`StratifiedShuffleSplit` (rather than `random_split`) preserves the class ratio in both train and validation:
- `train_split`: ~4,433 images (85 %)
- `val_split`  : ~783 images (15 %)

### WeightedRandomSampler
Used during backbone training so every mini-batch is approximately class-balanced, which stabilises Stage 1 / Stage 2 optimisation under the native 74 % / 26 % imbalance.

### Data loaders
| Loader            | Data                  | Purpose                                 |
|-------------------|-----------------------|-----------------------------------------|
| `train_loader`    | train split + aug     | Backbone training                       |
| `val_loader`      | val split  (no aug)   | Validation / threshold tuning           |
| `test_loader`     | test/      (no aug)   | Final evaluation (locked)               |
| `feat_*_loader`   | eval transform only   | Feature extraction for the ELM          |
"""))

cells.append(code(
'''_base_train = datasets.ImageFolder(data_root / "train", transform=train_transform)
_base_eval  = datasets.ImageFolder(data_root / "train", transform=eval_transform)
all_labels  = np.array([s[1] for s in _base_train.samples])
assert _base_train.classes == CLASS_NAMES

sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
train_idx, val_idx = next(sss.split(np.zeros(len(all_labels)), all_labels))
y_tr = all_labels[train_idx]
y_vl = all_labels[val_idx]

print(f"Stratified split (seed={SEED}):")
print(f"  train: {len(train_idx):,}  NORMAL={sum(y_tr==0)} PNEUMONIA={sum(y_tr==1)}")
print(f"  val  : {len(val_idx):,}  NORMAL={sum(y_vl==0)} PNEUMONIA={sum(y_vl==1)}")

# Save the split index so every run is auditable
all_paths = [s[0] for s in _base_eval.samples]
split_rows = []
for idx in train_idx:
    split_rows.append({"filepath": all_paths[idx], "label": CLASS_NAMES[all_labels[idx]], "split": "train"})
for idx in val_idx:
    split_rows.append({"filepath": all_paths[idx], "label": CLASS_NAMES[all_labels[idx]], "split": "val"})
for path, label_idx in datasets.ImageFolder(data_root / "test").samples:
    split_rows.append({"filepath": path, "label": CLASS_NAMES[label_idx], "split": "test"})
pd.DataFrame(split_rows).to_csv(SPLIT_CSV, index=False)
print(f"Split CSV saved: {SPLIT_CSV}")

train_split_ds = Subset(_base_train, train_idx)
val_split_ds   = Subset(_base_eval,  val_idx)
feat_tr_ds     = Subset(_base_eval,  train_idx)
test_ds        = datasets.ImageFolder(data_root / "test", transform=eval_transform)

# Oversample the minority class so every mini-batch is approximately balanced
sample_weights = [class_weights_np[l] for l in y_tr]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

_kw = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
           persistent_workers=(NUM_WORKERS > 0),
           prefetch_factor=2 if NUM_WORKERS > 0 else None)
train_loader   = DataLoader(train_split_ds, BATCH_SIZE, sampler=sampler, **_kw)
val_loader     = DataLoader(val_split_ds,   BATCH_SIZE, shuffle=False,   **_kw)
test_loader    = DataLoader(test_ds,        BATCH_SIZE, shuffle=False,   **_kw)
feat_tr_loader = DataLoader(feat_tr_ds,     BATCH_SIZE, shuffle=False,   **_kw)
feat_vl_loader = DataLoader(val_split_ds,   BATCH_SIZE, shuffle=False,   **_kw)
feat_te_loader = DataLoader(test_ds,        BATCH_SIZE, shuffle=False,   **_kw)

print(f"Loaders: train={len(train_split_ds)} val={len(val_split_ds)} test={len(test_ds)}")
'''))

# ──────────────────────────────────────────────────────────────────
# 8 — Model builder utilities
# ──────────────────────────────────────────────────────────────────
cells.append(md(
r"""## Cell 8 — Model builder & utility functions

- **`build_model()`** — instantiates EfficientNetV2-B0 via `timm` and moves it to the device.
- **`freeze_backbone()`** — freezes everything except the classifier head (used in Stage 1).
- **`unfreeze_last_fraction(f)`** — unfreezes the last $f \cdot 100\,\%$ of parameter tensors (used in Stage 2; e.g. 50 % ≈ 78 of the ~156 tensors).

### Feature dimensionality
After `forward_features` + `global_pool`, EfficientNetV2-B0 produces a **1,280-dimensional** vector per image — the input of the ELM hidden layer.
"""))

cells.append(code(
'''def build_model(pretrained=True):
    return timm.create_model(MODEL_NAME, pretrained=pretrained,
                             num_classes=NUM_CLASSES).to(DEVICE)

def trainable_stats(model):
    tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return tr, tot

def freeze_backbone(model):
    """Freeze every parameter except those named 'classifier.*' (Stage 1)."""
    for name, p in model.named_parameters():
        p.requires_grad = "classifier" in name

def unfreeze_last_fraction(model, fraction):
    """Keep classifier trainable and unfreeze the last `fraction` of parameter tensors (Stage 2)."""
    all_p = list(model.parameters())
    n_un  = max(1, int(len(all_p) * fraction))
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.requires_grad = True
    for p in all_p[-n_un:]:
        p.requires_grad = True
    return n_un

# Detect feature dim by running a dummy forward pass
_m = build_model(pretrained=False); _m.eval()
with torch.no_grad():
    _x = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    FEATURE_DIM = _m.global_pool(_m.forward_features(_x)).shape[1]
_n_tensors = len(list(_m.parameters()))
del _m

print(f"Feature dim: {FEATURE_DIM} | Param tensors: {_n_tensors}")
print(f"Unfreeze {STAGE2_FINETUNE_FRACTION*100:.0f}% = "
      f"{int(_n_tensors*STAGE2_FINETUNE_FRACTION)} tensors")
'''))

# ──────────────────────────────────────────────────────────────────
# 9 — Training utilities
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 9 — Training utilities (early stopping, train / eval loops)

- **`EarlyStopping`** — halts training when the validation metric has not improved for `patience` epochs; stores the best state so it can be restored after training.
- **`train_one_epoch`** — one training epoch with gradient clipping (`max_norm=1.0`) and `optimizer.zero_grad(set_to_none=True)` for lower memory usage.
- **`evaluate`** — inference loop returning loss, accuracy, predictions, and ground-truth labels.
"""))

cells.append(code(
'''class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.best_state = None
        self.counter = 0

    def step(self, score, model):
        better = (self.mode == "max" and (self.best_score is None or score > self.best_score)) or \\
                 (self.mode == "min" and (self.best_score is None or score < self.best_score))
        if better:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(model, loader, criterion, optimizer, epoch_idx, n_epochs):
    model.train()
    running_loss = correct = total = 0
    pbar = tqdm(loader, desc=f"  Ep {epoch_idx+1:02d}/{n_epochs}", leave=False, ncols=88)
    for imgs, labels in pbar:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        correct      += (logits.argmax(1) == labels).sum().item()
        total        += bs
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return running_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    crit = nn.CrossEntropyLoss()
    running_loss = correct = total = 0
    all_p, all_l = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        loss   = crit(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds   = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_p.extend(preds.cpu().numpy())
        all_l.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_p, all_l

print("Training utilities ready")
'''))

# ──────────────────────────────────────────────────────────────────
# 10 — Feature extraction helper
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 10 — Feature-extraction helper

Extracts backbone features that will feed the ELM:
1. `model.forward_features(x)` → feature maps.
2. `model.global_pool(...)`  → **1,280-D vector** per image.

Uses mixed-precision autocast on GPU for ~2× speed-up.

Returns `X` of shape `(N, 1280)` and `y` of shape `(N,)`.
"""))

cells.append(code(
'''@torch.inference_mode()
def extract_features(model, loader, desc=""):
    model.eval()
    all_f, all_l = [], []
    for imgs, labels in tqdm(loader, desc=f"  Extract {desc:10s}", leave=False, ncols=78):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            feats = model.forward_features(imgs)
            feats = model.global_pool(feats)
        all_f.append(feats.float().cpu())
        all_l.append(labels)
    return torch.cat(all_f), torch.cat(all_l)

print("Feature extraction function ready")
'''))

# ──────────────────────────────────────────────────────────────────
# 11 — Stage 1
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 11 — Stage 1: train the classifier head (backbone frozen)

> **First run only.** On subsequent runs, skip Cells 11–14 and execute Cell 15 (LOAD) instead to reuse the cached backbone and features.

### Strategy
- Freeze everything except the classifier (2 tensors: weight + bias).
- Loss: `CrossEntropyLoss(weight=class_weights, label_smoothing=0.10)`.
- Optimiser: AdamW, `lr=1e-3`, cosine-annealing schedule.
- Early stopping: `patience=4` on validation accuracy.

The head first learns the ImageNet → {Normal, Pneumonia} mapping without disturbing the pretrained backbone features.
"""))

cells.append(code(
'''model = build_model(pretrained=True)
freeze_backbone(model)
tr_p, tot_p = trainable_stats(model)
print(f"Stage 1 - Trainable: {tr_p:,}/{tot_p:,}  (head only)\\n")

criterion_s1 = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=STAGE1_LABEL_SMOOTH)
optimizer_s1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=STAGE1_LR, weight_decay=WEIGHT_DECAY)
scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_s1, T_max=STAGE1_EPOCHS, eta_min=STAGE1_LR * 0.01)
es1 = EarlyStopping(patience=STAGE1_PATIENCE, mode="max")

history_s1 = []
t_s1 = time.perf_counter()
for ep in range(STAGE1_EPOCHS):
    tr_l, tr_a       = train_one_epoch(model, train_loader, criterion_s1, optimizer_s1, ep, STAGE1_EPOCHS)
    vl_l, vl_a, _, _ = evaluate(model, val_loader)
    scheduler_s1.step()
    lr = scheduler_s1.get_last_lr()[0]
    stop = es1.step(vl_a, model)
    history_s1.append(dict(epoch=ep + 1, phase="stage1",
                           train_loss=tr_l, train_acc=tr_a,
                           val_loss=vl_l, val_acc=vl_a, lr=lr))
    tag = f"ES{es1.counter}/{STAGE1_PATIENCE}" if es1.counter > 0 else "best"
    print(f"  Ep {ep+1:2d}  loss={tr_l:.4f}  acc={tr_a:.4f}  "
          f"val_loss={vl_l:.4f}  val_acc={vl_a:.4f}  {tag}")
    if stop:
        print(f"  Early stop at epoch {ep+1}")
        break
t_stage1 = time.perf_counter() - t_s1
es1.load_best(model)
print(f"\\nStage 1 done - {t_stage1:.1f}s | Best val acc: {es1.best_score:.4f}")
'''))

# ──────────────────────────────────────────────────────────────────
# 12 — Stage 2
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 12 — Stage 2: partial fine-tuning (last 50 % unfrozen)

> **First run only.** Skip on subsequent runs.

### Strategy
- Unfreeze the last 50 % of parameter tensors (higher-level features).
- Lower learning rate `2e-5` to avoid catastrophic forgetting.
- Reduced label smoothing `0.05`.
- Early stopping with `patience=5`.

Higher-level features adapt to the chest X-ray domain while the low-level features (edges, textures) stay fixed.

After Stage 2 finishes, the best weights are written to `artifacts/best_backbone.pth`.
"""))

cells.append(code(
'''n_un = unfreeze_last_fraction(model, STAGE2_FINETUNE_FRACTION)
tr_p, tot_p = trainable_stats(model)
print(f"Stage 2 - Unfrozen {n_un} tensors "
      f"({STAGE2_FINETUNE_FRACTION*100:.0f}%)  Trainable: {tr_p:,}/{tot_p:,}\\n")

criterion_s2 = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=STAGE2_LABEL_SMOOTH)
optimizer_s2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=STAGE2_LR, weight_decay=WEIGHT_DECAY)
scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_s2, T_max=STAGE2_EPOCHS, eta_min=STAGE2_LR * 0.01)
es2 = EarlyStopping(patience=STAGE2_PATIENCE, mode="max")

history_s2 = []
t_s2 = time.perf_counter()
for ep in range(STAGE2_EPOCHS):
    tr_l, tr_a       = train_one_epoch(model, train_loader, criterion_s2, optimizer_s2, ep, STAGE2_EPOCHS)
    vl_l, vl_a, _, _ = evaluate(model, val_loader)
    scheduler_s2.step()
    lr = scheduler_s2.get_last_lr()[0]
    stop = es2.step(vl_a, model)
    tag = "best" if es2.counter == 0 else f"ES{es2.counter}/{STAGE2_PATIENCE}"
    history_s2.append(dict(epoch=ep + 1, phase="stage2",
                           train_loss=tr_l, train_acc=tr_a,
                           val_loss=vl_l, val_acc=vl_a, lr=lr))
    print(f"  Ep {ep+1:2d}  loss={tr_l:.4f}  acc={tr_a:.4f}  "
          f"val_loss={vl_l:.4f}  val_acc={vl_a:.4f}  {tag}")
    if stop:
        print(f"  Early stop at epoch {ep+1}")
        break
t_stage2         = time.perf_counter() - t_s2
t_backbone_total = t_stage1 + t_stage2
es2.load_best(model)

# Persist the best backbone weights
torch.save(model.state_dict(), BACKBONE_PATH)
print(f"\\nStage 2 done - {t_stage2:.1f}s | Best val acc: {es2.best_score:.4f}")
print(f"Backbone total: {t_backbone_total:.1f}s")
print(f"Weights saved: {BACKBONE_PATH}")
'''))

# ──────────────────────────────────────────────────────────────────
# 13 — Model 1 eval + feature save
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 13 — Model 1 evaluation + feature extraction + scaling

This cell performs three tasks in sequence:
1. **Evaluate Model 1** (EfficientNetV2-B0 + standard SGD head) on the locked test set and record timing.
2. **Extract 1,280-D features** for the train, validation, and test splits and save them as `features_v6.pt`.
3. **Fit `StandardScaler` on the training features**, transform all three splits, and save the result as `features_v6_scaled.npz`.

| File | Contents |
|------|----------|
| `features_v6.pt`        | Raw tensors: `X_train, y_train, X_val, y_val, X_test, y_test` |
| `features_v6_scaled.npz`| Scaled NumPy arrays + labels + scaler parameters + Model-1 predictions |
"""))

cells.append(code(
'''# ── Model 1 evaluation (Standard head) ──────────────────────────
@torch.inference_mode()
def predict_with_timing(model, loader):
    model.eval()
    all_p, all_l, all_pr = [], [], []
    # Warm-up batch to stabilise CUDA timing
    _wb = next(iter(loader))[0][:2].to(DEVICE); _ = model(_wb)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        probs  = torch.softmax(logits, 1)
        preds  = probs.argmax(1)
        all_p.extend(preds.cpu().numpy())
        all_l.extend(labels.numpy())
        all_pr.extend(probs[:, 1].cpu().numpy())
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (np.array(all_p), np.array(all_l), np.array(all_pr),
            time.perf_counter() - t0)

m1_preds, m1_labels, m1_probs, t_m1_inf = predict_with_timing(model, test_loader)
m1_acc = accuracy_score(m1_labels, m1_preds)
m1_prec, m1_rec, m1_f1, _ = precision_recall_fscore_support(
    m1_labels, m1_preds, average="binary", pos_label=1)
m1_auc = roc_auc_score(m1_labels, m1_probs)
y_test = m1_labels

print("=" * 55)
print("  Model 1: EfficientNetV2-B0 + Standard head")
print("=" * 55)
print(f"  Acc={m1_acc:.4f} Prec={m1_prec:.4f} Rec={m1_rec:.4f} "
      f"F1={m1_f1:.4f} AUC={m1_auc:.4f}")
print(classification_report(m1_labels, m1_preds, target_names=CLASS_NAMES))

# ── Feature extraction ──────────────────────────────────────────
print("Extracting features ...")
t_feat_start = time.perf_counter()
X_train_t, y_train_t = extract_features(model, feat_tr_loader, "train")
X_val_t,   y_val_t   = extract_features(model, feat_vl_loader, "val")
X_test_t,  y_test_t  = extract_features(model, feat_te_loader, "test")
if DEVICE == "cuda":
    torch.cuda.synchronize()
t_feat = time.perf_counter() - t_feat_start

# Save raw features (.pt)
torch.save({"X_train": X_train_t, "y_train": y_train_t,
            "X_val":   X_val_t,   "y_val":   y_val_t,
            "X_test":  X_test_t,  "y_test":  y_test_t}, FEATURES_FILE)
print(f"Raw features saved: {FEATURES_FILE}")

# Standardise features (fit on train only to prevent leakage)
X_train_np = X_train_t.numpy(); y_train    = y_train_t.numpy().astype(int)
X_val_np   = X_val_t.numpy();   y_val      = y_val_t.numpy().astype(int)
X_test_np  = X_test_t.numpy();  y_test_elm = y_test_t.numpy().astype(int)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_np)
X_val   = scaler.transform(X_val_np)
X_test  = scaler.transform(X_test_np)

np.savez(FEATURES_SCALED,
         X_train=X_train, y_train=y_train,
         X_val=X_val,     y_val=y_val,
         X_test=X_test,   y_test=y_test_elm,
         scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
         m1_preds=m1_preds, m1_probs=m1_probs, m1_labels=m1_labels)

print(f"Scaled features saved: {FEATURES_SCALED}")
print(f"Extraction time: {t_feat:.2f}s")
print(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
'''))

# ──────────────────────────────────────────────────────────────────
# 14 — Save history
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 14 — Save training history (first run only)

Writes Stage-1 / Stage-2 loss and accuracy curves and the timing summary. After this cell the cached artifacts (backbone + features + history + timing) are complete, so subsequent runs can jump straight to Cell 15.
"""))

cells.append(code(
'''hist = pd.DataFrame(history_s1 + history_s2)
hist["global_ep"] = range(1, len(hist) + 1)
hist.to_csv(HISTORY_CSV, index=False)

timing_first_run = {
    "t_stage1": round(t_stage1, 3),
    "t_stage2": round(t_stage2, 3),
    "t_backbone_total": round(t_backbone_total, 3),
    "t_feat": round(t_feat, 3),
    "t_m1_inf": round(t_m1_inf, 4),
    "stage1_epochs": len(history_s1),
    "stage2_epochs": len(history_s2),
}
with open(TIMING_JSON, "w") as f:
    json.dump(timing_first_run, f, indent=2)

print(f"Saved: {HISTORY_CSV}, {TIMING_JSON}")
print(f"Stage 1: {len(history_s1)} epochs  Stage 2: {len(history_s2)} epochs")
print("=" * 60)
print("  All artifacts saved - next run: skip Cells 11-14, use Cell 15")
print("=" * 60)
'''))

# ──────────────────────────────────────────────────────────────────
# 15 — LOAD cell (FIXED: no Kaggle hardcoded paths)
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 15 — LOAD cached backbone + features (second run onwards)

> Use this cell **instead of Cells 11–14** on subsequent runs to skip backbone training entirely.

### What this cell does
1. Loads the backbone weights from `artifacts/best_backbone.pth`.
2. Loads the pre-scaled features from `artifacts/feature_exports/features_v6_scaled.npz`.
3. Reconstructs the `StandardScaler` from the stored mean / scale.
4. Restores the Model-1 (standard head) predictions for later comparison.
5. Loads the timing data from `artifacts/timing_first_run.json`.

### Workflow
1. Run Cells 1–10 as usual (imports, config, data, loaders, CLAHE, utilities).
2. **Skip Cells 11–14.**
3. Run this cell (Cell 15).
4. Continue from Cell 16.
"""))

cells.append(code(
'''# ══════════════════════════════════════════════════════════════════
#  LOAD mode - use instead of Cells 11-14 when cached artifacts exist
# ══════════════════════════════════════════════════════════════════
assert BACKBONE_PATH.exists(),   f"Not found: {BACKBONE_PATH} - run Cells 11-14 first."
assert FEATURES_SCALED.exists(), f"Not found: {FEATURES_SCALED} - run Cells 11-14 first."

# 1. Load backbone
model = build_model(pretrained=False)
model.load_state_dict(torch.load(BACKBONE_PATH, map_location=DEVICE))
model.eval()
print(f"Backbone loaded: {BACKBONE_PATH}")

# 2. Load scaled features
data    = np.load(FEATURES_SCALED)
X_train = data["X_train"]; y_train    = data["y_train"]
X_val   = data["X_val"];   y_val      = data["y_val"]
X_test  = data["X_test"];  y_test_elm = data["y_test"]

# 3. Reconstruct the scaler (for downstream use if needed)
scaler = StandardScaler()
scaler.mean_          = data["scaler_mean"]
scaler.scale_         = data["scaler_scale"]
scaler.var_           = scaler.scale_ ** 2
scaler.n_features_in_ = len(scaler.mean_)

# 4. Restore Model 1 results
m1_preds  = data["m1_preds"]
m1_probs  = data["m1_probs"]
m1_labels = data["m1_labels"]
y_test    = m1_labels

m1_acc = accuracy_score(m1_labels, m1_preds)
m1_prec, m1_rec, m1_f1, _ = precision_recall_fscore_support(
    m1_labels, m1_preds, average="binary", pos_label=1)
m1_auc = roc_auc_score(m1_labels, m1_probs)
print(f"Model 1 restored: Acc={m1_acc:.4f} F1={m1_f1:.4f} AUC={m1_auc:.4f}")

# 5. Load timing (optional)
if TIMING_JSON.exists():
    with open(TIMING_JSON) as f:
        _t = json.load(f)
    t_stage1         = _t["t_stage1"]
    t_stage2         = _t["t_stage2"]
    t_backbone_total = _t["t_backbone_total"]
    t_feat           = _t["t_feat"]
    t_m1_inf         = _t["t_m1_inf"]
    history_s1       = [None] * _t["stage1_epochs"]
    history_s2       = [None] * _t["stage2_epochs"]
    print(f"Timing loaded: backbone={t_backbone_total:.1f}s  feat={t_feat:.1f}s")
else:
    t_stage1 = t_stage2 = t_backbone_total = t_feat = t_m1_inf = 0.0
    history_s1, history_s2 = [], []
    print("Warning: timing file not found, timing set to 0")

print(f"\\nFeatures: X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}")
print("=" * 60)
print("  LOAD COMPLETE - continue from Cell 16")
print("=" * 60)
'''))

# ──────────────────────────────────────────────────────────────────
# 16 — BilevelELM
# ──────────────────────────────────────────────────────────────────
cells.append(md(
r"""## Cell 16 — BilevelELM: DIPGM-VA (proposed), iBiG-SAM, aiBiG-SAM

### DIPGM-VA (proposed) — Algorithm 1 of the paper

**Step 1.** Viscosity step + double proximal update:

$$w_n = (1 - \alpha_n)x_n + \alpha_n\,(I - s\nabla\varphi)(x_n)$$

$$y_n = \mathrm{prox}_{c_n g}\!\bigl(w_n - c_n\nabla f(w_n)\bigr)$$

$$z_n = (1 - \gamma_n)x_n + \gamma_n\,\mathrm{prox}_{c_n g}\!\bigl(y_n - c_n\nabla f(y_n)\bigr)$$

**Step 2.** Double-inertial extrapolation on the $z$-sequence:

$$x_{n+1} = z_n + \theta_n(z_n - z_{n-1}) + \delta_n(z_{n-1} - z_{n-2})$$

**Key insight.** Applying inertial extrapolation to the post-proximal $z$-sequence (a "cleaned" signal) is more stable than applying it to the pre-proximal iterate, which is the source of the empirical convergence improvement over iBiG-SAM / aiBiG-SAM.

The lower-level problem is the L1-regularised least-squares fit (sparse feature selection); the upper-level objective is $\varphi(x)=\tfrac12\|x\|^2$ (structural risk minimisation).
"""))

cells.append(code(
r'''class BilevelELM:
    """
    ELM classifier whose output weights are solved by a bilevel optimisation solver.

    Solvers
    -------
    - 'dipgm_va'  : Double-Inertial Proximal Gradient Method with Viscosity Approximation (proposed).
    - 'ibig_sam'  : Single-inertial BiG-SAM.
    - 'aibig_sam' : Alternating-inertial BiG-SAM.

    Parameters
    ----------
    hidden_dim   : int    - number of hidden nodes L
    activation   : str    - 'relu' or 'tanh'
    reg_lambda   : float  - L1 regularisation (sparsity penalty)
    class_weight : str    - 'balanced' or None
    solver       : str    - 'dipgm_va', 'ibig_sam', or 'aibig_sam'
    max_iter     : int    - maximum solver iterations
    viscosity    : float  - viscosity / contraction parameter (shared across solvers for fair comparison)
    random_state : int    - seed for the random input weights
    """
    ACTIVATIONS = {"relu": lambda x: torch.relu(x), "tanh": lambda x: torch.tanh(x)}

    def __init__(self, hidden_dim=1024, activation="relu", reg_lambda=1e-5,
                 class_weight=None, solver="dipgm_va", max_iter=500,
                 viscosity=0.01, random_state=42):
        self.hidden_dim   = hidden_dim
        self.activation   = activation
        self.reg_lambda   = reg_lambda
        self.class_weight = class_weight
        self.solver       = solver
        self.max_iter     = max_iter
        self.viscosity    = viscosity
        self.random_state = random_state
        self.W_in = self.b_in = self.beta = self.classes_ = None
        self.history_ = []
        self.device   = DEVICE
        self.n_iter_  = 0

    def _init_weights(self, input_dim):
        gen = torch.Generator().manual_seed(self.random_state)
        scale = (2.0 / (input_dim + self.hidden_dim)) ** 0.5
        self.W_in = torch.randn(self.hidden_dim, input_dim,
                                generator=gen, dtype=torch.float32).to(self.device) * scale
        self.b_in = torch.randn(1, self.hidden_dim,
                                generator=gen, dtype=torch.float32).to(self.device) * scale

    def _hidden(self, X):
        return self.ACTIVATIONS[self.activation](X @ self.W_in.T + self.b_in)

    @staticmethod
    def _soft_threshold(x, tau):
        # Proximal operator of the L1 norm (element-wise soft thresholding)
        return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0)

    def _prox_grad(self, point, Hw, Tw, c_n):
        # One proximal-gradient step for f(x) = ||Hx - T||^2
        grad_f = 2.0 * Hw.T @ (Hw @ point - Tw)
        return self._soft_threshold(point - c_n * grad_f, c_n * self.reg_lambda)

    def _objective(self, Hw, Tw, beta):
        res = Hw @ beta - Tw
        return (res * res).sum().item() + self.reg_lambda * beta.abs().sum().item()

    # ──────────────────────────────────────────────────────────────
    #  DIPGM-VA (proposed) - Algorithm 1
    #    Step 1:  w_n = (1 - alpha_n) x_n + alpha_n (I - s nabla phi)(x_n)
    #             y_n = prox_{c_n g}(w_n - c_n nabla f(w_n))
    #             z_n = (1 - gamma_n) x_n + gamma_n prox_{c_n g}(y_n - c_n nabla f(y_n))
    #    Step 2:  x_{n+1} = z_n + theta_n (z_n - z_{n-1}) + delta_n (z_{n-1} - z_{n-2})
    # ──────────────────────────────────────────────────────────────
    def _solve_dipgm_va(self, Hw, Tw, Lf):
        D, C = Hw.shape[1], Tw.shape[1]
        dev  = self.device
        s    = 0.5
        z_prev2 = torch.zeros(D, C, dtype=torch.float32, device=dev)
        z_prev1 = torch.zeros(D, C, dtype=torch.float32, device=dev)
        x_curr  = torch.zeros(D, C, dtype=torch.float32, device=dev)
        self.history_ = []
        for n in range(1, self.max_iter + 1):
            alpha_n = 1.0 / (n + 1)
            gamma_n = 0.9 * n / (n + 1)
            tau_n   = 1e14 / (n * n)
            mu_n    = n / (n + 1)
            rho_n   = 1.0 / (n ** 2)
            c_n     = 1.0 / Lf
            # Step 1 (nabla phi(x) = x since phi(x) = 1/2 ||x||^2)
            w_n     = (1.0 - alpha_n * s) * x_curr
            y_n     = self._prox_grad(w_n, Hw, Tw, c_n)
            prox_y  = self._prox_grad(y_n, Hw, Tw, c_n)
            z_curr  = (1.0 - gamma_n) * x_curr + gamma_n * prox_y
            # Step 2: double-inertial extrapolation on the z-sequence
            diff1   = z_curr - z_prev1
            diff2   = z_prev1 - z_prev2
            n1 = torch.norm(diff1).item()
            n2 = torch.norm(diff2).item()
            theta_n = min(mu_n,  alpha_n * tau_n / n1) if n1 > 1e-12 else mu_n
            delta_n = max(-rho_n, -alpha_n * tau_n / n2) if n2 > 1e-12 else -rho_n
            x_new   = z_curr + theta_n * diff1 + delta_n * diff2
            z_prev2 = z_prev1.clone()
            z_prev1 = z_curr.clone()
            x_curr  = x_new
            if n % 10 == 0 or n == self.max_iter:
                obj = self._objective(Hw, Tw, x_curr)
                self.history_.append({"iter": n, "objective": obj})
                if len(self.history_) >= 2 and abs(self.history_[-1]["objective"] - self.history_[-2]["objective"]) < 1e-8:
                    self.n_iter_ = n
                    return x_curr
        self.n_iter_ = self.max_iter
        return x_curr

    # ──────────────────────────────────────────────────────────────
    #  iBiG-SAM - single-inertial BiG-SAM baseline (Shehu et al.)
    # ──────────────────────────────────────────────────────────────
    def _solve_ibig_sam(self, Hw, Tw, Lf):
        D, C = Hw.shape[1], Tw.shape[1]
        dev  = self.device
        lambda_u = self.viscosity
        c        = 1.0 / Lf
        alpha_p  = 3.0
        gamma_base = 0.2 / (1.0 - (2.0 + c * Lf) / 4.0)
        x_prev = torch.zeros(D, C, dtype=torch.float32, device=dev)
        x_curr = torch.zeros(D, C, dtype=torch.float32, device=dev)
        self.history_ = []
        for n in range(1, self.max_iter + 1):
            gamma_n  = gamma_base
            beta_seq = gamma_n * (n ** (-0.01))
            diff     = x_curr - x_prev
            nd       = torch.norm(diff).item()
            bt       = n / (n + alpha_p - 1.0)
            theta_n  = min(bt, beta_seq / nd) if nd > 1e-12 else bt
            y_n      = x_curr + theta_n * diff
            p_n      = self._prox_grad(y_n, Hw, Tw, c)
            q_n      = (1.0 - lambda_u) * y_n
            x_new    = gamma_n * q_n + (1.0 - gamma_n) * p_n
            x_prev   = x_curr.clone()
            x_curr   = x_new
            if n % 10 == 0 or n == self.max_iter:
                obj = self._objective(Hw, Tw, x_curr)
                self.history_.append({"iter": n, "objective": obj})
                if len(self.history_) >= 2 and abs(self.history_[-1]["objective"] - self.history_[-2]["objective"]) < 1e-8:
                    self.n_iter_ = n
                    return x_curr
        self.n_iter_ = self.max_iter
        return x_curr

    # ──────────────────────────────────────────────────────────────
    #  aiBiG-SAM - alternating-inertial BiG-SAM baseline
    # ──────────────────────────────────────────────────────────────
    def _solve_aibig_sam(self, Hw, Tw, Lf):
        D, C = Hw.shape[1], Tw.shape[1]
        dev  = self.device
        lambda_u = self.viscosity
        c        = 1.0 / Lf
        alpha_p  = 3.0
        gamma_base = 0.2 / (1.0 - (2.0 + c * Lf) / 4.0)
        x_prev = torch.zeros(D, C, dtype=torch.float32, device=dev)
        x_curr = torch.zeros(D, C, dtype=torch.float32, device=dev)
        self.history_ = []
        for n in range(1, self.max_iter + 1):
            gamma_n  = gamma_base
            beta_seq = gamma_n * (n ** (-0.01))
            diff     = x_curr - x_prev
            nd       = torch.norm(diff).item()
            bt       = n / (n + alpha_p - 1.0)
            theta_n  = min(bt, beta_seq / nd) if nd > 1e-12 else bt
            # Apply the inertial term only on odd iterations
            y_n      = (x_curr + theta_n * diff) if n % 2 == 1 else x_curr.clone()
            p_n      = self._prox_grad(y_n, Hw, Tw, c)
            q_n      = (1.0 - lambda_u) * y_n
            x_new    = gamma_n * q_n + (1.0 - gamma_n) * p_n
            x_prev   = x_curr.clone()
            x_curr   = x_new
            if n % 10 == 0 or n == self.max_iter:
                obj = self._objective(Hw, Tw, x_curr)
                self.history_.append({"iter": n, "objective": obj})
                if len(self.history_) >= 2 and abs(self.history_[-1]["objective"] - self.history_[-2]["objective"]) < 1e-8:
                    self.n_iter_ = n
                    return x_curr
        self.n_iter_ = self.max_iter
        return x_curr

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_c = len(self.classes_)
        N, D = X.shape
        self._init_weights(D)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        H   = self._hidden(X_t)
        T   = torch.zeros(N, n_c, dtype=torch.float32, device=self.device)
        for i, c in enumerate(self.classes_):
            T[:, i] = torch.tensor(y == c, dtype=torch.float32)
        # Inverse-frequency class weighting (balanced)
        if self.class_weight == "balanced":
            counts = np.bincount(y.astype(int), minlength=n_c)
            sqrt_w = torch.tensor(
                np.sqrt(N / (n_c * np.maximum(counts, 1)))[y.astype(int)].astype(np.float32)
            ).to(self.device)
        else:
            sqrt_w = torch.ones(N, dtype=torch.float32, device=self.device)
        Hw = sqrt_w.unsqueeze(1) * H
        Tw = sqrt_w.unsqueeze(1) * T
        # Lipschitz constant of the lower-level gradient (2 * lambda_max(H^T H))
        Lf = 2.0 * torch.linalg.eigvalsh(Hw.T @ Hw)[-1].item()
        if   self.solver == "dipgm_va":  self.beta = self._solve_dipgm_va(Hw, Tw, Lf)
        elif self.solver == "ibig_sam":  self.beta = self._solve_ibig_sam(Hw, Tw, Lf)
        elif self.solver == "aibig_sam": self.beta = self._solve_aibig_sam(Hw, Tw, Lf)
        return self

    def predict_proba(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        return torch.softmax(self._hidden(X_t) @ self.beta, dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def sparsity_ratio(self):
        return (self.beta.abs() < 1e-8).float().mean().item() if self.beta is not None else 0.0

    def n_active_nodes(self):
        return (self.beta.abs().sum(dim=1) > 1e-8).sum().item() if self.beta is not None else 0

print("BilevelELM: DIPGM-VA (proposed), iBiG-SAM, aiBiG-SAM")
'''))

# ──────────────────────────────────────────────────────────────────
# 17 — Grid search
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 17 — Grid search on DIPGM-VA

The best configuration selected by DIPGM-VA on the validation set is then reused by iBiG-SAM and aiBiG-SAM to guarantee a fair comparison (same hidden size, same L1 regularisation, same random seed).
"""))

cells.append(code(
'''n_gs = len(BILEVEL_HIDDEN_GRID) * len(BILEVEL_REG_GRID) * len(BILEVEL_ACT_GRID)
print(f"Grid search (DIPGM-VA): {n_gs} combos")
gs_rows = []
best_f1 = -1
best_p  = {}
i       = 0
t_gs0   = time.perf_counter()
for h in BILEVEL_HIDDEN_GRID:
    for reg in BILEVEL_REG_GRID:
        for act in BILEVEL_ACT_GRID:
            i += 1
            elm = BilevelELM(hidden_dim=h, activation=act, reg_lambda=reg,
                             class_weight=ELM_CLASS_WEIGHT, solver="dipgm_va",
                             max_iter=BILEVEL_MAX_ITER, random_state=SEED)
            elm.fit(X_train, y_train)
            vp = elm.predict(X_val)
            va = accuracy_score(y_val, vp)
            vpr, vre, vf, _ = precision_recall_fscore_support(
                y_val, vp, average="binary", pos_label=1, zero_division=0)
            gs_rows.append(dict(hidden_dim=h, reg_lambda=reg, activation=act,
                                val_acc=va, val_f1=vf,
                                sparsity=elm.sparsity_ratio(), n_iter=elm.n_iter_))
            is_best = vf > best_f1
            if is_best:
                best_f1 = vf
                best_p  = dict(hidden_dim=h, reg_lambda=reg, activation=act)
            print(f"  [{i:2d}/{n_gs}] h={h:4d} lam={reg:.1e} {act:4s}  "
                  f"acc={va:.4f} f1={vf:.4f} sp={elm.sparsity_ratio():.1%} "
                  f"it={elm.n_iter_}{' ***' if is_best else ''}")
t_gs  = time.perf_counter() - t_gs0
gs_df = pd.DataFrame(gs_rows).sort_values("val_f1", ascending=False)
print(f"\\nBest: {best_p} | F1={best_f1:.4f} | {t_gs:.1f}s")
try:
    display(gs_df.head(5).round(4))
except NameError:
    print(gs_df.head(5).round(4))
'''))

# ──────────────────────────────────────────────────────────────────
# 18 — Final models
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 18 — Train the final models

Train DIPGM-VA, iBiG-SAM, and aiBiG-SAM with the same best hyperparameters identified above, so the only variable between runs is the solver.
"""))

cells.append(code(
'''print(f"Final models: {best_p}\\n")
models_info = []
for solver, label in [("dipgm_va", "DIPGM-VA"),
                      ("ibig_sam", "iBiG-SAM"),
                      ("aibig_sam", "aiBiG-SAM")]:
    t0 = time.perf_counter()
    m  = BilevelELM(**best_p, class_weight=ELM_CLASS_WEIGHT,
                    solver=solver, max_iter=BILEVEL_MAX_ITER, random_state=SEED)
    m.fit(X_train, y_train)
    tt = time.perf_counter() - t0
    models_info.append({"label": label, "model": m, "time": tt})
    print(f"  {label:10s}: {tt*1000:.1f}ms  iter={m.n_iter_}  sp={m.sparsity_ratio():.1%}")

final_dipgm = models_info[0]["model"]; t_dipgm = models_info[0]["time"]
final_ibig  = models_info[1]["model"]; t_ibig  = models_info[1]["time"]
final_aibig = models_info[2]["model"]; t_aibig = models_info[2]["time"]

print(f"\\nSpeedup vs Stage 1 backbone training ({t_stage1:.1f}s):")
for mi in models_info:
    print(f"  {mi['label']:10s}: {t_stage1 / max(mi['time'], 1e-9):,.0f}x")
'''))

# ──────────────────────────────────────────────────────────────────
# 19 — Threshold tuning
# ──────────────────────────────────────────────────────────────────
cells.append(md(
r"""## Cell 19 — Threshold tuning via Youden's J statistic (on the validation split)

For each ELM, we sweep thresholds in $[0.05, 0.96]$ and pick the one that maximises

$$J = \text{sensitivity} + \text{specificity} - 1$$

on the validation set. This yields a balanced operating point before we ever touch the test set.
"""))

cells.append(code(
'''def tune_threshold(elm, Xv, yv):
    probs = elm.predict_proba(Xv)[:, 1]
    best_j = -1
    best_t = 0.5
    for ti in range(50, 960, 5):
        thr = ti / 1000.0
        p   = (probs >= thr).astype(int)
        cm  = confusion_matrix(yv, p)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        j  = se + sp - 1
        if j > best_j:
            best_j = j
            best_t = round(float(thr), 3)
    return best_t

thr_dipgm = tune_threshold(final_dipgm, X_val, y_val)
thr_ibig  = tune_threshold(final_ibig,  X_val, y_val)
thr_aibig = tune_threshold(final_aibig, X_val, y_val)
print(f"Thresholds: DIPGM-VA={thr_dipgm}  iBiG-SAM={thr_ibig}  aiBiG-SAM={thr_aibig}")
'''))

# ──────────────────────────────────────────────────────────────────
# 20 — Test evaluation
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 20 — Test-set evaluation (four models)

Evaluates all four models on the **locked test set** (624 images). The Youden-tuned thresholds from Cell 19 are applied.
"""))

cells.append(code(
'''def eval_elm(elm, Xt, yt, thr, name):
    pr  = elm.predict_proba(Xt)[:, 1]
    pd_ = (pr >= thr).astype(int)
    a   = accuracy_score(yt, pd_)
    p, r, f, _ = precision_recall_fscore_support(yt, pd_, average="binary", pos_label=1)
    auc = roc_auc_score(yt, pr)
    cm  = confusion_matrix(yt, pd_)
    sp  = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape == (2, 2) else 0
    return dict(name=name, acc=a, prec=p, rec=r, spec=sp, f1=f, auc=auc,
                bacc=(r + sp) / 2, threshold=thr, preds=pd_, probs=pr)

print("=" * 60)
print("  TEST SET (n=624)")
print("=" * 60)

cm1   = confusion_matrix(m1_labels, m1_preds)
m1_sp = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
r1 = dict(name="Std Head", acc=m1_acc, prec=m1_prec, rec=m1_rec, spec=m1_sp,
          f1=m1_f1, auc=m1_auc, bacc=(m1_rec + m1_sp) / 2,
          threshold=0.5, preds=m1_preds, probs=m1_probs)
r2 = eval_elm(final_dipgm, X_test, y_test, thr_dipgm, "DIPGM-VA")
r3 = eval_elm(final_ibig,  X_test, y_test, thr_ibig,  "iBiG-SAM")
r4 = eval_elm(final_aibig, X_test, y_test, thr_aibig, "aiBiG-SAM")
all_results = [r1, r2, r3, r4]

for r in all_results:
    print(f"  {r['name']:10s}: Acc={r['acc']:.4f} Prec={r['prec']:.4f} "
          f"Rec={r['rec']:.4f} F1={r['f1']:.4f} AUC={r['auc']:.4f}")

results_df = pd.DataFrame(
    [{k: v for k, v in r.items() if k not in ("preds", "probs")} for r in all_results])
try:
    display(results_df.round(4))
except NameError:
    print(results_df.round(4))
'''))

# ──────────────────────────────────────────────────────────────────
# 21 — Convergence
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 21 — Convergence analysis

Plots the lower-level objective value per iteration for the three bilevel solvers. DIPGM-VA typically reaches the plateau in fewer iterations, which is the empirical basis of the paper's speed-up claim.
"""))

cells.append(code(
'''fig, axes = plt.subplots(1, 2, figsize=(15, 5))
cols  = {"DIPGM-VA": "#C44E52", "iBiG-SAM": "#4C72B0", "aiBiG-SAM": "#55A868"}
marks = {"DIPGM-VA": "o",       "iBiG-SAM": "s",       "aiBiG-SAM": "^"}
for mi in models_info:
    h = pd.DataFrame(mi["model"].history_)
    if len(h) > 0:
        axes[0].semilogy(h["iter"], h["objective"],
                         f'{marks[mi["label"]]}-', ms=3, lw=1.5,
                         color=cols[mi["label"]],
                         label=f'{mi["label"]} (iter={mi["model"].n_iter_})')
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Objective (log scale)")
axes[0].set_title("Convergence of bilevel ELM solvers", fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[1].axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_convergence.png")
plt.show()
print(f"Saved: {FIG_DIR}/fig_convergence.png")
'''))

# ──────────────────────────────────────────────────────────────────
# 22 — Confusion matrices
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 22 — Confusion matrices (all four models)
"""))

cells.append(code(
'''fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, r in zip(axes, all_results):
    cm = confusion_matrix(y_test, r["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f'{r["name"]}\\nacc={r["acc"]:.4f}', fontsize=9)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
plt.suptitle("Confusion matrices (n=624)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_confusion_matrices.png")
plt.show()
print(f"Saved: {FIG_DIR}/fig_confusion_matrices.png")
'''))

# ──────────────────────────────────────────────────────────────────
# 23 — Final table
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 23 — Final comparison table

Reproduces Table X of the paper: accuracy, precision, recall, specificity, F1, AUC, balanced accuracy, threshold, classifier training time, speed-up, and iteration count for all four models. Best value per row is marked with `*`.
"""))

cells.append(code(
'''times = {"Std Head": t_stage1, "DIPGM-VA": t_dipgm,
         "iBiG-SAM": t_ibig,  "aiBiG-SAM": t_aibig}
iters = {"Std Head": "N/A",
         "DIPGM-VA": str(final_dipgm.n_iter_),
         "iBiG-SAM": str(final_ibig.n_iter_),
         "aiBiG-SAM": str(final_aibig.n_iter_)}
names = [r["name"] for r in all_results]

print("\\n" + "=" * 80)
print("  FINAL COMPARISON TABLE")
print("=" * 80)
hdr = f"{'Metric':<16}" + "".join(f"{n:>14}" for n in names)
print(hdr)
print("-" * len(hdr))

for metric, key in [("Accuracy", "acc"), ("Precision", "prec"),
                    ("Recall", "rec"), ("Specificity", "spec"),
                    ("F1-score", "f1"), ("AUC-ROC", "auc"),
                    ("Balanced Acc", "bacc")]:
    vals = [r[key] for r in all_results]
    bi   = int(np.argmax(vals))
    row  = f"  {metric:<14}"
    for i, v in enumerate(vals):
        row += f"  {v:>10.4f}{'*' if i == bi else ' '}"
    print(row)

print("-" * len(hdr))
row = f"  {'Threshold':<14}"
for n in names:
    row += f"  {all_results[names.index(n)]['threshold']:>11}"
print(row)

row = f"  {'Clf time':<14}"
for n in names:
    t = times[n]
    row += f"  {t:>8.2f}s  " if t >= 1 else f"  {t*1000:>8.1f}ms "
print(row)

row = f"  {'Speedup':<14}"
for n in names:
    if n == "Std Head":
        row += f"  {'1x':>11} "
    else:
        row += f"  {t_stage1 / max(times[n], 1e-9):>10.0f}x "
print(row)

row = f"  {'Iterations':<14}"
for n in names:
    row += f"  {iters[n]:>11}"
print(row)

print("=" * 80)
print("  * = best | Backbone shared across all four models | Threshold: Youden's J")
'''))

# ──────────────────────────────────────────────────────────────────
# 24 — Training curves
# ──────────────────────────────────────────────────────────────────
cells.append(md(
"""## Cell 24 — Figure 3: backbone training curves

Plots the Stage-1 and Stage-2 loss / accuracy curves stored in `training_history.csv`. Requires that Cells 11–14 ran at least once (otherwise the file does not exist and this figure is skipped).
"""))

cells.append(code(
'''hist_path = HISTORY_CSV
if hist_path.exists():
    hist = pd.read_csv(hist_path)
    if "global_ep" not in hist.columns:
        hist["global_ep"] = range(1, len(hist) + 1)
    n_s1 = len(hist[hist["phase"] == "stage1"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Loss
    axes[0].plot(hist["global_ep"], hist["train_loss"], "o-", ms=4,
                 label="Train loss", color="#0072B2", lw=1.5)
    axes[0].plot(hist["global_ep"], hist["val_loss"], "s-", ms=4,
                 label="Val loss",   color="#D55E00", lw=1.5)
    axes[0].axvline(x=n_s1 + 0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ymin, ymax = axes[0].get_ylim()
    axes[0].text(n_s1 * 0.5, ymax * 0.92, "Stage 1\\n(Head only)",
                 ha="center", fontsize=9, color="gray", style="italic")
    axes[0].text(n_s1 + (len(hist) - n_s1) * 0.5, ymax * 0.92,
                 "Stage 2\\n(Fine-tune 50%)",
                 ha="center", fontsize=9, color="gray", style="italic")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("(a) Training & validation loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) Accuracy
    axes[1].plot(hist["global_ep"], hist["train_acc"], "o-", ms=4,
                 label="Train acc", color="#0072B2", lw=1.5)
    axes[1].plot(hist["global_ep"], hist["val_acc"], "s-", ms=4,
                 label="Val acc",   color="#D55E00", lw=1.5)
    axes[1].axvline(x=n_s1 + 0.5, color="gray", ls="--", lw=1, alpha=0.7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("(b) Training & validation accuracy", fontweight="bold")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Fig. 3: EfficientNetV2-B0 backbone training curves",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_training_curves.png")
    plt.savefig(FIG_DIR / "fig3_training_curves.pdf")
    plt.show()
    print(f"Saved: {FIG_DIR}/fig3_training_curves.png / .pdf")
else:
    print(f"WARNING: {hist_path} not found - run Cells 11-14 first to produce this figure.")
'''))

# ──────────────────────────────────────────────────────────────────
# Write notebook
# ──────────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {DST}  ({len(cells)} cells)")
