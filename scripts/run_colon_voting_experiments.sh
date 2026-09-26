#!/usr/bin/env bash
set -euo pipefail

run_bundle() {
  model="$1"
  c_value="$2"
  regime="$3"
  bundle="results/voting_experiments_v1/colon_cancer/${model}/${regime}"

  poetry run python scripts/run_cv_predictions.py \
    --data data/colon_cancer.csv \
    --base-model "$model" \
    --C "$c_value" \
    --n-neurons 31 \
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

run_bundle svm 1.3 low
run_bundle svm 5.2 high
run_bundle cpl 2.0 low
run_bundle cpl 8.0 high
run_bundle logreg 3.4 low
run_bundle logreg 5.2 high
