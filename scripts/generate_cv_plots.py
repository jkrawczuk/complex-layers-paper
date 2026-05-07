#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edu.but.experiments.cv_plots import (
    save_accuracy_plot,
    save_available_features_plot,
    save_feature_recovery_plot,
    save_features_plot,
    save_loss_l1_plot,
    save_minority_metrics_plot,
    save_precision_recall_trajectory_plot,
    save_train_test_gap_plot,
)


_RESULTS_KEY_MAP = {
    "vote_accuracy": "mean_accuracy",
    "vote_accuracy_std": "std_accuracy",
    "vote_balanced_accuracy": "mean_balanced_accuracy",
    "vote_balanced_accuracy_std": "std_balanced_accuracy",
    "vote_minority_precision": "mean_minority_precision",
    "vote_minority_precision_std": "std_minority_precision",
    "vote_minority_recall": "mean_minority_recall",
    "vote_minority_recall_std": "std_minority_recall",
    "vote_minority_f1": "mean_minority_f1",
    "vote_minority_f1_std": "std_minority_f1",
    "vote_train_accuracy": "mean_train_accuracy",
    "vote_train_accuracy_std": "std_train_accuracy",
    "vote_train_balanced_accuracy": "mean_train_balanced_accuracy",
    "vote_train_balanced_accuracy_std": "std_train_balanced_accuracy",
    "vote_train_minority_precision": "mean_train_minority_precision",
    "vote_train_minority_precision_std": "std_train_minority_precision",
    "vote_train_minority_recall": "mean_train_minority_recall",
    "vote_train_minority_recall_std": "std_train_minority_recall",
    "vote_train_minority_f1": "mean_train_minority_f1",
    "vote_train_minority_f1_std": "std_train_minority_f1",
    "features_nth_neuron": "features",
    "features_nth_neuron_std": "features_std",
}


def _parse_csv(path: Path, rename_keys: bool):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                out_key = _RESULTS_KEY_MAP.get(key, key) if rename_keys else key
                if value == "":
                    parsed[out_key] = None
                    continue
                try:
                    parsed[out_key] = float(value)
                except ValueError:
                    parsed[out_key] = value
            rows.append(parsed)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from computed CV metrics.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--n-total-features", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_dir = args.bundle_dir / "metrics"
    figures_dir = args.bundle_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = _parse_csv(metrics_dir / "results.csv", rename_keys=True)
    per_split_rows = _parse_csv(metrics_dir / "per_split_metrics.csv", rename_keys=False)
    weighted_results_path = metrics_dir / "weighted_vote_results.csv"
    weighted_results = None
    if weighted_results_path.exists():
        weighted_results = _parse_csv(weighted_results_path, rename_keys=True)

    save_accuracy_plot(results, figures_dir / "accuracy.png", weighted_results=weighted_results)
    save_loss_l1_plot(results, figures_dir / "loss_l1_norm.png")
    save_features_plot(results, figures_dir / "features_nth_neuron.png")
    save_train_test_gap_plot(results, figures_dir / "neuron_train_test_accuracy.png")
    save_minority_metrics_plot(results, figures_dir / "minority_metrics.png")
    save_precision_recall_trajectory_plot(results, figures_dir / "precision_recall_trajectory.png")
    save_available_features_plot(
        per_split_rows,
        args.n_total_features,
        figures_dir / "available_features_per_neuron.png",
    )
    save_feature_recovery_plot(results, figures_dir / "feature_recovery.png")
    print(f"saved figures dir: {figures_dir}")


if __name__ == "__main__":
    main()
