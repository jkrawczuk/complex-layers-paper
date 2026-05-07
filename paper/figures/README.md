# Manuscript Figures

This directory contains the final raster figure assets included by `paper/main.tex`.

Current figure sets:

- `colon_cancer/` - sparse-regime accuracy and feature-count panels for `cpl`, `svm`, and `logreg`.
- `synthetic_signal/` - sparse-regime accuracy and ground-truth feature-recovery panels for `cpl`, `svm`, and `logreg`.

The files were generated from local experiment bundles under `results/paper_runs/` using:

```bash
poetry run python scripts/generate_cv_plots.py \
  --bundle-dir <bundle-dir> \
  --n-total-features 2000
```

The `results/` directory is ignored by git. To regenerate these figures from scratch, first run the
three-step experiment pipeline described in the repository-level `README.md`, then copy the relevant
`figures/*.png` outputs into the dataset-specific directories here.
