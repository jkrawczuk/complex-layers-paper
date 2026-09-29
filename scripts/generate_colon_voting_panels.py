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
REGIMES = ("low", "high", "high_top5")
REGIME_LABELS = {"low": "low", "high": "high", "high_top5": "high-top5"}
MODEL_LABELS = {"cpl": "CPL", "svm": "SVM", "logreg": "Logistic regression"}
COLORS = {
    "uniform": "#1f77b4",
    "train_balanced_accuracy_minus_chance": "#2ca02c",
    "inverse_l2_norm": "#ff7f0e",
    "sample_distance": "#d62728",
    "inverse_l1_norm": "#9467bd",
    "sample_distance_linf": "#8c564b",
    "train": "#1f77b4",
    "test": "#ff7f0e",
}
LABELS = {
    "uniform": "uniform",
    "train_balanced_accuracy_minus_chance": "train BA",
    "inverse_l2_norm": r"inverse $\|w\|_2$",
    "sample_distance": r"sample distance $L_2$",
    "inverse_l1_norm": r"inverse $\|w\|_1$",
    "sample_distance_linf": r"sample distance $L_\infty$",
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


def grouped_ratio(rows, numerator_key, denominator_key, factor=1.0):
    values = defaultdict(list)
    for row in rows:
        numerator = row.get(numerator_key)
        denominator = row.get(denominator_key)
        if numerator not in (None, "") and denominator not in (None, ""):
            denominator = float(denominator)
            if denominator > 0.0:
                values[int(row["n_neuron"])].append(
                    factor * float(numerator) / denominator
                )
    return values


def median_iqr(values):
    x = np.asarray(values, dtype=float)
    return float(np.median(x)), float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))


def setup_axes(title, ylabel):
    fig, axes = plt.subplots(len(REGIMES), 3, figsize=(15, 12.2), sharex=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.93, wspace=0.22, hspace=0.38)
    fig.suptitle(title, fontsize=16, y=0.98)
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            ax.set_title(f"{MODEL_LABELS[model]} — {REGIME_LABELS[regime]}")
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


def margin_width_linf_panel():
    fig, axes = setup_axes(
        r"Colon cancer: $L_\infty$ unit-score band width across neuron depth",
        r"median $2 / \|w\|_1$ (IQR)",
    )
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            rows = read_csv(RESULTS / model / regime / "neuron_summary.csv")
            grouped = defaultdict(list)
            for item in rows:
                norm = float(item["l1_norm"])
                if norm > 0.0:
                    grouped[int(item["n_neuron"])].append(2.0 / norm)
            plot_median_band(axes[row, col], grouped, "band width", COLORS["inverse_l1_norm"])
            axes[row, col].set_yscale("log")
            axes[row, col].text(
                0.98, 0.04, "bands: IQR across available CV splits",
                transform=axes[row, col].transAxes, ha="right", va="bottom", fontsize=8, alpha=0.7,
            )
    path = OUT / "colon_margin_width_linf_panel.png"
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


def geometric_margin_linf_panel():
    fig, axes = setup_axes(
        r"Colon cancer: signed $L_\infty$ margin across neuron depth",
        r"median signed margin $y f(x) / \|w\|_1$ (IQR)",
    )
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            rows = read_csv(RESULTS / model / regime / "neuron_summary.csv")
            ax = axes[row, col]
            plot_median_band(
                ax, grouped_ratio(rows, "train_functional_margin_median", "l1_norm"),
                "train", COLORS["train"],
            )
            plot_median_band(
                ax, grouped_ratio(rows, "test_functional_margin_median", "l1_norm"),
                "test", COLORS["test"],
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.legend(fontsize=8)
    path = OUT / "colon_geometric_margin_linf_panel.png"
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


def paired_accuracy_difference_panel():
    fig, axes = setup_axes(
        "Colon cancer: paired test-accuracy difference vs uniform voting",
        r"mean paired $\Delta$ accuracy (95% bootstrap interval)",
    )
    compared = (
        "train_balanced_accuracy_minus_chance",
        "inverse_l2_norm",
        "sample_distance",
        "inverse_l1_norm",
        "sample_distance_linf",
    )
    rng = np.random.default_rng(20260927)
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            rows = read_csv(
                RESULTS / model / regime / "metrics" / "voting_strategy_per_split_metrics.csv"
            )
            by_strategy = defaultdict(dict)
            for item in rows:
                key = (int(item["split_id"]), int(item["n_neurons"]))
                by_strategy[item["weight_metric"]][key] = float(item["vote_accuracy"])
            ax = axes[row, col]
            baseline = by_strategy["uniform"]
            for strategy in compared:
                grouped = defaultdict(list)
                for key, value in by_strategy[strategy].items():
                    if key in baseline:
                        grouped[key[1]].append(value - baseline[key])
                depth = np.asarray(sorted(grouped), dtype=int)
                means = []
                lows = []
                highs = []
                for n in depth:
                    delta = np.asarray(grouped[n], dtype=float)
                    means.append(float(np.mean(delta)))
                    if delta.size == 1:
                        lows.append(float(delta[0]))
                        highs.append(float(delta[0]))
                    else:
                        samples = rng.choice(delta, size=(5000, delta.size), replace=True).mean(axis=1)
                        lows.append(float(np.quantile(samples, 0.025)))
                        highs.append(float(np.quantile(samples, 0.975)))
                ax.plot(
                    depth, means, color=COLORS[strategy], marker="o", markersize=2.5,
                    linewidth=1.4, label=LABELS[strategy],
                )
                ax.fill_between(depth, lows, highs, color=COLORS[strategy], alpha=0.10)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.legend(fontsize=7)
    path = OUT / "colon_paired_accuracy_difference_panel.png"
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


def feature_count_panel():
    fig, axes = setup_axes(
        "Colon cancer: features selected by successive neurons",
        "median selected features (IQR)",
    )
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            rows = read_csv(RESULTS / model / regime / "neuron_summary.csv")
            plot_median_band(
                axes[row, col], grouped_values(rows, "features_nth_neuron"),
                "selected features", COLORS["uniform"],
            )
            axes[row, col].set_ylim(bottom=0.0)
    path = OUT / "colon_selected_features_panel.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def coverage_panel():
    fig, axes = setup_axes(
        "Colon cancer: CV-split availability across neuron depth",
        "available CV splits (maximum 100)",
    )
    for row, regime in enumerate(REGIMES):
        for col, model in enumerate(MODELS):
            rows = read_csv(RESULTS / model / regime / "neuron_summary.csv")
            counts = defaultdict(int)
            for item in rows:
                counts[int(item["n_neuron"])] += 1
            depth = np.asarray(sorted(counts), dtype=int)
            coverage = np.asarray([counts[n] for n in depth], dtype=int)
            ax = axes[row, col]
            ax.plot(depth, coverage, color=COLORS["uniform"], marker="o", markersize=3)
            ax.fill_between(depth, 0, coverage, color=COLORS["uniform"], alpha=0.10)
            ax.axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_ylim(0, 105)
    path = OUT / "colon_split_coverage_panel.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (
        margin_width_panel(),
        margin_width_linf_panel(),
        geometric_margin_panel(),
        geometric_margin_linf_panel(),
        voting_panel(),
        paired_accuracy_difference_panel(),
        feature_count_panel(),
        coverage_panel(),
        timing_panel(),
    ):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
