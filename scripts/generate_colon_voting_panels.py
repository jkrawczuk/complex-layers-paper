#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "voting_experiments_v1" / "colon_cancer"
OUT = RESULTS / "figures"
MODELS = ("cpl", "svm", "logreg")
REGIMES = ("low", "high")
MODEL_LABELS = {"cpl": "CPL", "svm": "SVM", "logreg": "Logistic regression"}
COLORS = {
    "uniform": "#1f77b4",
    "train_balanced_accuracy_minus_chance": "#2ca02c",
    "inverse_l2_norm": "#ff7f0e",
    "sample_distance": "#d62728",
    "train": "#1f77b4",
    "test": "#ff7f0e",
}
LABELS = {
    "uniform": "uniform",
    "train_balanced_accuracy_minus_chance": "train BA",
    "inverse_l2_norm": r"inverse $\|w\|_2$",
    "sample_distance": "sample distance",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grouped_values(rows, key):
    values = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            values[int(row["n_neuron"])].append(float(value))
    return values


def median_iqr(values):
    x = np.asarray(values, dtype=float)
    return float(np.median(x)), float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))


def setup_axes(title, ylabel):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.90, wspace=0.22, hspace=0.32)
    fig.suptitle(title, fontsize=16, y=0.97)
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            ax.set_title(f"{MODEL_LABELS[model]} — {regime}")
            ax.set_xlabel("neuron depth")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=0.22)
    return fig, axes


def plot_median_band(ax, grouped, label, color, linestyle="-"):
    depth = np.asarray(sorted(grouped), dtype=int)
    stats = [median_iqr(grouped[n]) for n in depth]
    median = np.asarray([v[0] for v in stats])
    low = np.asarray([v[1] for v in stats])
    high = np.asarray([v[2] for v in stats])
    ax.plot(depth, median, color=color, linestyle=linestyle, marker="o", markersize=3, label=label)
    ax.fill_between(depth, low, high, color=color, alpha=0.13)


def margin_width_panel():
    fig, axes = setup_axes(
        r"Colon cancer: unit-score band width across neuron depth",
        r"median $2 / \|w\|_2$ (IQR)",
    )
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            rows = read_csv(RESULTS / model / regime / "neuron_summary.csv")
            grouped = grouped_values(rows, "unit_score_band_width")
            plot_median_band(axes[row, col], grouped, "band width", COLORS["uniform"])
            axes[row, col].set_yscale("log")
            axes[row, col].text(
                0.98, 0.04, "bands: IQR across available CV splits",
                transform=axes[row, col].transAxes, ha="right", va="bottom", fontsize=8, alpha=0.7,
            )
    path = OUT / "colon_margin_width_panel.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def geometric_margin_panel():
    fig, axes = setup_axes(
        "Colon cancer: signed geometric margin across neuron depth",
        "median signed geometric margin (IQR)",
    )
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            rows = read_csv(RESULTS / model / regime / "neuron_summary.csv")
            ax = axes[row, col]
            plot_median_band(
                ax, grouped_values(rows, "train_geometric_margin_median"),
                "train", COLORS["train"],
            )
            plot_median_band(
                ax, grouped_values(rows, "test_geometric_margin_median"),
                "test", COLORS["test"],
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.legend(fontsize=8)
    path = OUT / "colon_geometric_margin_panel.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def voting_panel():
    fig, axes = setup_axes(
        "Colon cancer: voting strategies across neuron depth",
        "mean test accuracy",
    )
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            rows = read_csv(
                RESULTS / model / regime / "metrics" / "voting_strategy_results.csv"
            )
            ax = axes[row, col]
            for strategy in LABELS:
                selected = sorted(
                    (r for r in rows if r["weight_metric"] == strategy),
                    key=lambda r: int(r["n_neurons"]),
                )
                ax.plot(
                    [int(r["n_neurons"]) for r in selected],
                    [float(r["vote_accuracy"]) for r in selected],
                    color=COLORS[strategy], marker="o", markersize=2.5,
                    linewidth=1.4, label=LABELS[strategy],
                )
            ax.legend(fontsize=7)
    path = OUT / "colon_voting_strategy_panel.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def timing_panel():
    fig, axes = setup_axes(
        "Colon cancer: fit time of successive neurons",
        "median fit time, seconds (IQR)",
    )
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            rows = read_csv(RESULTS / model / regime / "neuron_summary.csv")
            plot_median_band(
                axes[row, col], grouped_values(rows, "fit_time_sec"),
                "fit time", COLORS["uniform"],
            )
    path = OUT / "colon_neuron_fit_time_panel.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (
        margin_width_panel(),
        geometric_margin_panel(),
        voting_panel(),
        timing_panel(),
    ):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
