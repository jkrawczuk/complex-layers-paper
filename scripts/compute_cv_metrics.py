#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edu.but.experiments.cv_metrics import compute_cv_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Compute CV metrics from raw prediction bundles.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    compute_cv_metrics(parse_args().bundle_dir)


if __name__ == "__main__":
    main()
