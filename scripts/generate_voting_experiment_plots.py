#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_mean_std(rows, group_key, value_key):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value not in (None, ""):
            grouped[row[group_key]].append(float(value))
    out = {}
    for key, values in grouped.items():
        out[key] = (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    return out


def _plot_series(ax, rows, value_key, label, color=None, linestyle="-"):
    stats = _group_mean_std(rows, "n_neuron", value_key)
    x = np.asarray(sorted(int(key) for key in stats), dtype=int)
    mean = np.asarray([stats[str(n)][0] for n in x])
    std = np.asarray([stats[str(n)][1] for n in x])
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, marker="o", markersize=3)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)


def save_margin_diagnostics(bundle_dir, out_dir):
    rows = _read_csv(bundle_dir / "neuron_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    _plot_series(axes[0, 0], rows, "inverse_l2_norm", r"$1/\|w\|_2$")
    _plot_series(axes[0, 0], rows, "unit_score_band_width", r"$2/\|w\|_2$")
    axes[0, 0].set_title("Coefficient-norm width diagnostics")
    axes[0, 0].set_ylabel("width")
    axes[0, 0].legend()

    _plot_series(axes[0, 1], rows, "train_geometric_margin_median", "train")
    _plot_series(axes[0, 1], rows, "test_geometric_margin_median", "test")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Median signed geometric margin")
    axes[0, 1].legend()

    _plot_series(axes[1, 0], rows, "train_nonpositive_margin_fraction", "train")
    _plot_series(axes[1, 0], rows, "test_nonpositive_margin_fraction", "test")
    axes[1, 0].set_title("Fraction on the wrong side of the boundary")
    axes[1, 0].set_ylabel("fraction")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].legend()

    _plot_series(axes[1, 1], rows, "fit_time_sec", "fit time")
    axes[1, 1].set_title("Fit time per neuron")
    axes[1, 1].set_ylabel("seconds")

    for ax in axes.flat:
        ax.set_xlabel("neuron depth")
        ax.grid(alpha=0.2)

    path = out_dir / "margin_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_voting_comparison(bundle_dir, out_dir):
    rows = _read_csv(bundle_dir / "metrics" / "voting_strategy_results.csv")
    by_strategy = defaultdict(list)
    for row in rows:
        by_strategy[row["weight_metric"]].append(row)

    labels = {
        "uniform": "uniform",
        "train_balanced_accuracy_minus_chance": "train BA",
        "inverse_l2_norm": "inverse L2 norm",
        "sample_distance": "sample distance",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for strategy, strategy_rows in sorted(by_strategy.items()):
        strategy_rows.sort(key=lambda row: int(row["n_neurons"]))
        x = [int(row["n_neurons"]) for row in strategy_rows]
        accuracy = [float(row["vote_accuracy"]) for row in strategy_rows]
        accuracy_std = [float(row["vote_accuracy_std"]) for row in strategy_rows]
        line = axes[0].plot(x, accuracy, marker="o", markersize=3, label=labels.get(strategy, strategy))[0]
        axes[0].fill_between(
            x,
            np.asarray(accuracy) - np.asarray(accuracy_std),
            np.asarray(accuracy) + np.asarray(accuracy_std),
            color=line.get_color(),
            alpha=0.1,
        )
        axes[1].plot(
            x,
            [float(row["aggregation_time_sec"]) for row in strategy_rows],
            marker="o",
            markersize=3,
            label=labels.get(strategy, strategy),
        )

    axes[0].set_title("Voting accuracy")
    axes[0].set_ylabel("accuracy")
    axes[1].set_title("Voting aggregation time")
    axes[1].set_ylabel("seconds")
    for ax in axes:
        ax.set_xlabel("number of neurons")
        ax.grid(alpha=0.2)
        ax.legend()

    path = out_dir / "voting_strategy_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot margin and voting-strategy diagnostics.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.bundle_dir / "figures" / "voting_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"saved: {save_margin_diagnostics(args.bundle_dir, out_dir)}")
    print(f"saved: {save_voting_comparison(args.bundle_dir, out_dir)}")


if __name__ == "__main__":
    main()
