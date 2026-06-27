#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edu.but.experiments.cv_run import run_cv_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Run CV and save raw per-neuron predictions.")
    parser.add_argument("--data", type=Path, default=Path("data/colon_cancer.csv"))
    parser.add_argument("--label-first", action="store_true")
    parser.add_argument("--has-header", action="store_true")
    parser.add_argument("--n-neurons", type=int, default=31)
    parser.add_argument(
        "--base-model",
        choices=["logreg", "svm", "cpl"],
        default="cpl",
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "uniform"],
        default="balanced",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--standardize-features",
        action="store_true",
        help="Standardize feature columns within each CV split using training-fold statistics.",
    )
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    run_cv_predictions(parse_args())


if __name__ == "__main__":
    main()
