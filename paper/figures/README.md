# Figures README

This directory stores final figure assets used in the manuscript and standalone composite figure layouts.

## Structure

- `paper/figures/<dataset>/<C>/`:
  final per-run figure files copied or prepared for the paper
- `paper/figures/<dataset>/composites/`:
  standalone composite layouts such as `5x4` grids

Current datasets with prepared figure sets:
- `colon_cancer`
- `colon_5`

## 5x4 Grid Convention

The `5x4` grids are comparison sheets built from CPL-LP runs across four regularization settings.

Columns:
- `C=3`
- `C=1`
- `C=0.3`
- `C=0.1`

Equivalent lambda labels used in headers:
- `lambda = 1/(3m)`
- `lambda = 1/m`
- `lambda = 3/m`
- `lambda = 10/m`

Rows:
- `accuracy`
- `neuron_train_test_accuracy`
- `loss_l1_norm`
- `features_nth_neuron`
- `available_features_per_neuron`

## Source Results

Composite figures read source plots directly from `results/`, for example:

- `results/colon_cancer/cpl_lp_n200/balanced/C3/`
- `results/colon_cancer/cpl_lp_n200/balanced/C1/`
- `results/colon_cancer/cpl_lp_n200/balanced/C0p3/`
- `results/colon_cancer/cpl_lp_n200/balanced/C0p1/`
- `results/colon_5/cpl_lp_n200/balanced/C3/`
- `results/colon_5/cpl_lp_n200/balanced/C1/`
- `results/colon_5/cpl_lp_n200/balanced/C0p3/`
- `results/colon_5/cpl_lp_n200/balanced/C0p1/`

## Existing Composite Files

Examples already present in the repo:

- `paper/figures/colon_cancer/composites/colon_cancer_5x4_grid_lambda_highres_balanced.pdf`
- `paper/figures/colon_cancer/composites/colon_cancer_5x4_grid_lambda_highres_uniform.pdf`
- `paper/figures/colon_5/composites/colon_5_5x4_grid_lambda_highres_balanced.pdf`

## Rebuild

To rebuild a standalone composite PDF:

```bash
cd paper/figures/<dataset>/composites
pdflatex -interaction=nonstopmode <composite_file>.tex
```

Example:

```bash
cd paper/figures/colon_5/composites
pdflatex -interaction=nonstopmode colon_5_5x4_grid_lambda_highres_balanced.tex
```
