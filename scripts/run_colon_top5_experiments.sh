#!/usr/bin/env bash
set -euo pipefail

run_bundle() {
  model="$1"
  c_value="$2"
  bundle="results/voting_experiments_v1/colon_cancer/${model}/high_top5"

  poetry run python scripts/run_cv_predictions.py \
    --data data/colon_cancer.csv \
    --base-model "$model" \
    --C "$c_value" \
    --n-neurons 31 \
    --top-k-features 5 \
    --cv-folds 10 \
    --cv-repeats 10 \
    --jobs 1 \
    --standardize-features \
    --verbose 0 \
    --out-dir "$bundle"

  poetry run python scripts/compute_cv_metrics.py --bundle-dir "$bundle"
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/complex-layers-matplotlib-cache \
    poetry run python scripts/generate_voting_experiment_plots.py \
    --bundle-dir "$bundle"
}

run_bundle svm 5.2
run_bundle cpl 8.0
run_bundle logreg 5.2
