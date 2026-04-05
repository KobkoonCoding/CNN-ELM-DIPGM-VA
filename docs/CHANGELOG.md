# Changelog — `Code.ipynb` → `Code_clean.ipynb`

Rebuilt the notebook from scratch (via `build_clean_notebook.py`) to prepare it
for public release alongside the IEEE Access manuscript. Original file
`Code.ipynb` is **unchanged** and kept as a historical reference.

## Summary of edits

### Language
- **All Thai markdown cells translated to English** and rewritten in the same
  terminology as the manuscript (DIPGM-VA, EfficientNetV2-B0, Kermany dataset,
  CLAHE, Youden's J, bilevel optimisation).
- All Thai inline comments in code cells translated to English.
- Long decorative Unicode separators kept where they aid readability,
  non-ASCII arrows (`→`, `←`, `⚠`) replaced with ASCII equivalents in `print`
  statements so output renders correctly on Windows consoles.

### Structural / reproducibility fixes
- **Single `CONFIG` block** at the top (Cell 3): every hyperparameter, path,
  and seed is defined in one place.
- **Artifacts relocated** under `./artifacts/` (was scattered:
  `best_backbone.pth`, `feature_exports/features_v6.pt`,
  `feature_exports/features_v6_scaled.npz`, `training_history.csv`,
  `timing_first_run.json` are all now under `artifacts/`).
- **Deterministic seeding** added: `random`, `numpy`, `torch`, `torch.cuda`.
- **Device auto-detection**: replaces the implicit Tesla-P100 assumption.
- **Kaggle-only hardcoded paths removed from the LOAD cell** (was
  `/kaggle/input/datasets/feveroliang/my-pretrained-files/...`). The LOAD cell
  now reads the same `CONFIG` paths as the first-run cells, so the notebook is
  portable between local, Colab, and Kaggle kernels.

### Dead / scratch code removed
- Deleted the obsolete `LASSO_*` configuration block and the leftover
  "FIX 1" comment (unused — never referenced after the bilevel grid was
  introduced).
- Removed duplicate imports (matplotlib / seaborn / numpy / pandas were
  re-imported in cell 11 of the original).
- Removed the stale `"EfficientNetV2-B0 + Lasso-ELM (ISTA/FISTA)"` header in
  the config print — replaced with the correct `"Bilevel ELM (DIPGM-VA)"`.
- Removed orphaned `display()` calls that assume an IPython environment; they
  are now wrapped in a `try/except NameError` fallback to `print`.

### Comments & docstrings
- `BilevelELM` class now has a proper English docstring that describes the
  three solvers and their shared parameters.
- One-line comments added above: soft-threshold operator, proximal-gradient
  step, Step 1 / Step 2 of DIPGM-VA, Lipschitz-constant computation,
  class-weighted Stage-1 training, and the StandardScaler fit (with a note
  that it is fit on training data only to avoid leakage).
- Mathematical formulas in markdown now use standard LaTeX (no Unicode-only
  operators) so they render correctly on GitHub and in Jupyter.

### Print statements
- Verified that every remaining `print(...)` uses correct f-string syntax and
  references variables that exist at that point in the pipeline.
- Removed debug prints that reported internal intermediate state with no
  reader value.

### Cell count
- Original notebook: 50 cells (25 markdown + 25 code).
- Cleaned notebook: 51 cells (25 markdown + 25 code + 1 extra markdown header
  introducing the plot-style cell that was previously un-labelled).

## New files produced
- `Code_clean.ipynb` — the cleaned notebook.
- `requirements.txt` — pinned Python dependencies.
- `environment.yml` — optional Conda environment.
- `build_clean_notebook.py` — the generator script (kept in-repo so the clean
  notebook is itself reproducible).
- `CHANGELOG_clean.md` — this file.
