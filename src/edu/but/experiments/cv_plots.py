from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_DPI = 180
PLOT_FIGSIZE = (9.6, 6.4)

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 17,
    }
)


def _set_sparse_xticks(ax, x_vals):
    if not x_vals:
        return
    if len(x_vals) <= 20:
        ticks = x_vals
    else:
        step = 5
        ticks = list(range(int(min(x_vals)), int(max(x_vals)) + 1, step))
        if int(max(x_vals)) not in ticks:
            ticks.append(int(max(x_vals)))
    ax.set_xticks(ticks)


def _ci95_bounds(mean_vals, std_vals, coverage_vals, low_clip=None, high_clip=None):
    lows = []
    highs = []
    for m, s, c in zip(mean_vals, std_vals, coverage_vals):
        if c is None or c <= 0:
            half = 0.0
        else:
            half = 1.96 * (float(s) / np.sqrt(float(c)))
        lo = float(m) - half
        hi = float(m) + half
        if low_clip is not None:
            lo = max(low_clip, lo)
        if high_clip is not None:
            hi = min(high_clip, hi)
        lows.append(lo)
        highs.append(hi)
    return lows, highs


def save_accuracy_plot(results, out_path: Path, weighted_results=None):
    x = [r["n_neurons"] for r in results]
    vote = [r["mean_accuracy"] for r in results]
    vote_std = [r["std_accuracy"] for r in results]
    neuron = [r["neuron_accuracy"] for r in results]
    neuron_std = [r["neuron_accuracy_std"] for r in results]
    coverage = [r.get("coverage_n") for r in results]
    has_coverage = all(v is not None for v in coverage)
    if not has_coverage:
        coverage = [1 for _ in x]

    if has_coverage:
        fig, (ax, ax_cov) = plt.subplots(
            2,
            1,
            figsize=PLOT_FIGSIZE,
            dpi=PLOT_DPI,
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0]},
        )
    else:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
        ax_cov = None
    vote_lo, vote_hi = _ci95_bounds(vote, vote_std, coverage, low_clip=0.0, high_clip=1.0)
    neuron_lo, neuron_hi = _ci95_bounds(neuron, neuron_std, coverage, low_clip=0.0, high_clip=1.0)
    ax.plot(x, vote, marker="o", linewidth=2.0, color="#1f77b4", label="vote_accuracy")
    ax.plot(x, neuron, marker="s", linewidth=2.0, color="#d62728", label="neuron_accuracy")
    ax.fill_between(
        x,
        vote_lo,
        vote_hi,
        color="#1f77b4",
        alpha=0.12,
    )
    ax.fill_between(
        x,
        neuron_lo,
        neuron_hi,
        color="#d62728",
        alpha=0.12,
    )
    if weighted_results:
        weighted_x = [r["n_neurons"] for r in weighted_results]
        weighted_vote = [r["mean_accuracy"] for r in weighted_results]
        weighted_vote_std = [r["std_accuracy"] for r in weighted_results]
        weighted_coverage = [r.get("coverage_n") for r in weighted_results]
        if not all(v is not None for v in weighted_coverage):
            weighted_coverage = [1 for _ in weighted_x]
        weighted_lo, weighted_hi = _ci95_bounds(
            weighted_vote,
            weighted_vote_std,
            weighted_coverage,
            low_clip=0.0,
            high_clip=1.0,
        )
        ax.plot(
            weighted_x,
            weighted_vote,
            marker="^",
            linestyle="--",
            linewidth=2.0,
            color="#2ca02c",
            label="weighted_vote_accuracy",
        )
        ax.fill_between(
            weighted_x,
            weighted_lo,
            weighted_hi,
            color="#2ca02c",
            alpha=0.10,
        )
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    ax.set_title("Accuracy: mean lines with 95% CI bands")
    if has_coverage and ax_cov is not None:
        ax_cov.plot(x, coverage, marker="o", linewidth=1.8, color="#444444", label="coverage_n")
        ax_cov.set_ylabel("coverage")
        ax_cov.set_xlabel("k (neuron index)")
        ax_cov.grid(True, alpha=0.35)
        ax_cov.set_title("Coverage_n: number of CV splits per neuron")
        _set_sparse_xticks(ax_cov, x)
    else:
        ax.set_xlabel("k (neuron index)")
        _set_sparse_xticks(ax, x)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_loss_l1_plot(results, out_path: Path):
    x = [r["n_neurons"] for r in results]
    loss = [r["loss"] for r in results]
    loss_std = [r["loss_std"] for r in results]
    l1 = [r["l1_norm"] for r in results]
    l1_std = [r["l1_norm_std"] for r in results]

    fig, ax_loss = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    ax_l1 = ax_loss.twinx()

    loss_line = ax_loss.plot(x, loss, marker="o", linewidth=2.0, color="#2ca02c", label="loss")[0]
    l1_line = ax_l1.plot(x, l1, marker="s", linewidth=2.0, color="#ff7f0e", label="l1_norm")[0]

    ax_loss.fill_between(
        x,
        [a - s for a, s in zip(loss, loss_std)],
        [a + s for a, s in zip(loss, loss_std)],
        color="#2ca02c",
        alpha=0.12,
    )
    ax_l1.fill_between(
        x,
        [a - s for a, s in zip(l1, l1_std)],
        [a + s for a, s in zip(l1, l1_std)],
        color="#ff7f0e",
        alpha=0.12,
    )

    ax_loss.set_xlabel("k (neuron index)")
    ax_loss.set_ylabel("loss", color="#2ca02c")
    ax_l1.set_ylabel("l1_norm", color="#ff7f0e")
    ax_loss.tick_params(axis="y", colors="#2ca02c")
    ax_l1.tick_params(axis="y", colors="#ff7f0e")
    _set_sparse_xticks(ax_loss, x)
    ax_loss.grid(True, alpha=0.35)
    ax_loss.legend([loss_line, l1_line], ["loss", "l1_norm"], frameon=True, loc="best")
    ax_loss.set_title("Loss/L1: mean lines with +/-1 std bands")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_features_plot(results, out_path: Path):
    x = [r["n_neurons"] for r in results]
    features = [r["features"] for r in results]
    features_std = [r["features_std"] for r in results]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    ax.plot(
        x,
        features,
        marker="o",
        linewidth=2.0,
        color="#9467bd",
        label="features_nth_neuron",
    )
    ax.fill_between(
        x,
        [max(0.0, a - s) for a, s in zip(features, features_std)],
        [a + s for a, s in zip(features, features_std)],
        color="#9467bd",
        alpha=0.12,
    )
    ax.set_xlabel("k (neuron index)")
    ax.set_ylabel("features_nth_neuron")
    _set_sparse_xticks(ax, x)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    ax.set_title("Features per neuron: mean line with +/-1 std band")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_available_features_plot(per_split_rows, n_total_features: int, out_path: Path):
    remaining_by_n = defaultdict(list)
    rows_by_split = defaultdict(list)
    for row in per_split_rows:
        rows_by_split[row["split_id"]].append(row)

    for split_rows in rows_by_split.values():
        split_rows_sorted = sorted(split_rows, key=lambda r: int(r["n_neurons"]))
        remaining = int(n_total_features)
        for row in split_rows_sorted:
            n = int(row["n_neurons"])
            remaining_by_n[n].append(float(remaining))
            remaining -= int(row["features_nth_neuron"])
            if remaining < 0:
                remaining = 0

    x = sorted(remaining_by_n.keys())
    y_mean = [float(np.mean(remaining_by_n[n])) for n in x]
    y_std = [float(np.std(remaining_by_n[n])) for n in x]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    ax.plot(
        x,
        y_mean,
        marker="o",
        linewidth=2.0,
        color="#444444",
        label="available features before neuron k",
    )
    ax.fill_between(
        x,
        [max(0.0, m - s) for m, s in zip(y_mean, y_std)],
        [m + s for m, s in zip(y_mean, y_std)],
        color="#444444",
        alpha=0.12,
    )
    ax.set_xlabel("k (neuron index)")
    ax.set_ylabel("available features")
    ax.set_ylim(bottom=0)
    _set_sparse_xticks(ax, x)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    ax.set_title("Available features before neuron k: mean line with +/-1 std band")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_train_test_gap_plot(results, out_path: Path):
    x = [r["n_neurons"] for r in results]
    coverage = [r.get("coverage_n") for r in results]
    has_coverage = all(v is not None for v in coverage)
    if not has_coverage:
        coverage = [1 for _ in x]

    neuron_test = [r["neuron_accuracy"] for r in results]
    neuron_test_std = [r["neuron_accuracy_std"] for r in results]
    neuron_train = [r["neuron_train_accuracy"] for r in results]
    neuron_train_std = [r["neuron_train_accuracy_std"] for r in results]
    train_lo, train_hi = _ci95_bounds(neuron_train, neuron_train_std, coverage, low_clip=0.0, high_clip=1.0)
    test_lo, test_hi = _ci95_bounds(neuron_test, neuron_test_std, coverage, low_clip=0.0, high_clip=1.0)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    ax.plot(
        x,
        neuron_train,
        marker="o",
        linewidth=2.0,
        color="#2ca02c",
        label="neuron train accuracy",
    )
    ax.plot(
        x,
        neuron_test,
        marker="s",
        linewidth=2.0,
        color="#d62728",
        label="neuron test accuracy",
    )
    ax.fill_between(x, train_lo, train_hi, color="#2ca02c", alpha=0.12)
    ax.fill_between(x, test_lo, test_hi, color="#d62728", alpha=0.12)
    ax.set_xlabel("k (neuron index)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.5, 1.0)
    _set_sparse_xticks(ax, x)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    ax.set_title("Neuron train/test accuracy: mean lines with 95% CI bands")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_minority_metrics_plot(results, out_path: Path):
    x = [r["n_neurons"] for r in results]
    coverage = [r.get("coverage_n") for r in results]
    has_coverage = all(v is not None for v in coverage)
    if not has_coverage:
        coverage = [1 for _ in x]

    metric_specs = [
        (
            "balanced accuracy",
            "mean_balanced_accuracy",
            "std_balanced_accuracy",
            "neuron_balanced_accuracy",
            "neuron_balanced_accuracy_std",
        ),
        (
            "minority precision",
            "mean_minority_precision",
            "std_minority_precision",
            "neuron_minority_precision",
            "neuron_minority_precision_std",
        ),
        (
            "minority recall",
            "mean_minority_recall",
            "std_minority_recall",
            "neuron_minority_recall",
            "neuron_minority_recall_std",
        ),
        (
            "minority F1",
            "mean_minority_f1",
            "std_minority_f1",
            "neuron_minority_f1",
            "neuron_minority_f1_std",
        ),
    ]

    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(9.6, 12.4), dpi=PLOT_DPI, sharex=True)
    colors = {"vote": "#1f77b4", "neuron": "#d62728"}

    for ax, (ylabel, vote_key, vote_std_key, neuron_key, neuron_std_key) in zip(axes, metric_specs):
        vote = [r[vote_key] for r in results]
        vote_std = [r[vote_std_key] for r in results]
        neuron = [r[neuron_key] for r in results]
        neuron_std = [r[neuron_std_key] for r in results]

        vote_lo, vote_hi = _ci95_bounds(vote, vote_std, coverage, low_clip=0.0, high_clip=1.0)
        neuron_lo, neuron_hi = _ci95_bounds(
            neuron, neuron_std, coverage, low_clip=0.0, high_clip=1.0
        )

        ax.plot(x, vote, marker="o", linewidth=2.0, color=colors["vote"], label="vote")
        ax.plot(x, neuron, marker="s", linewidth=2.0, color=colors["neuron"], label="neuron")
        ax.fill_between(x, vote_lo, vote_hi, color=colors["vote"], alpha=0.12)
        ax.fill_between(x, neuron_lo, neuron_hi, color=colors["neuron"], alpha=0.12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.35)

    axes[0].legend(frameon=True, loc="best")
    axes[0].set_title("Minority-class metrics: mean lines with 95% CI bands")
    axes[-1].set_xlabel("k (neuron index)")
    _set_sparse_xticks(axes[-1], x)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_precision_recall_trajectory_plot(results, out_path: Path):
    vote_precision = np.asarray([r["mean_minority_precision"] for r in results], dtype=float)
    vote_recall = np.asarray([r["mean_minority_recall"] for r in results], dtype=float)
    neuron_precision = np.asarray([r["neuron_minority_precision"] for r in results], dtype=float)
    neuron_recall = np.asarray([r["neuron_minority_recall"] for r in results], dtype=float)
    k_vals = np.asarray([r["n_neurons"] for r in results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), dpi=PLOT_DPI, sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")

    panels = [
        ("vote", vote_recall, vote_precision),
        ("neuron", neuron_recall, neuron_precision),
    ]

    for ax, (title, recall_vals, precision_vals) in zip(axes, panels):
        ax.plot(recall_vals, precision_vals, color="#666666", linewidth=1.4, alpha=0.8)
        scatter = ax.scatter(
            recall_vals,
            precision_vals,
            c=k_vals,
            cmap=cmap,
            s=28,
            edgecolors="none",
        )
        if len(k_vals) > 0:
            ax.annotate("k=1", (recall_vals[0], precision_vals[0]), xytext=(6, 6), textcoords="offset points", fontsize=8)
            ax.annotate(
                f"k={int(k_vals[-1])}",
                (recall_vals[-1], precision_vals[-1]),
                xytext=(6, -10),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(f"Minority precision vs recall ({title})")
        ax.set_xlabel("minority recall")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.35)

    axes[0].set_ylabel("minority precision")
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.95)
    cbar.set_label("k (neuron index)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_feature_recovery_plot(results, out_path: Path):
    if not results or results[0].get("cumulative_feature_recall") is None:
        return

    x = [r["n_neurons"] for r in results]
    coverage = [r.get("coverage_n") for r in results]
    if not all(v is not None for v in coverage):
        coverage = [1 for _ in x]

    fig, axes = plt.subplots(3, 1, figsize=(9.6, 11.4), dpi=PLOT_DPI, sharex=True)
    panels = [
        (
            axes[0],
            "cumulative recall",
            "cumulative_feature_recall",
            "cumulative_feature_recall_std",
            "cumulative_feature_weight_recall",
            "cumulative_feature_weight_recall_std",
            "feature recall",
            "weighted recall",
        ),
        (
            axes[1],
            "per-neuron purity",
            "feature_precision_nth",
            "feature_precision_nth_std",
            "noise_fraction_nth",
            "noise_fraction_nth_std",
            "informative precision",
            "noise fraction",
        ),
        (
            axes[2],
            "selected features",
            "informative_selected_nth",
            "informative_selected_nth_std",
            "noise_selected_nth",
            "noise_selected_nth_std",
            "informative selected",
            "noise selected",
        ),
    ]

    colors = ("#1f77b4", "#d62728")
    for ax, title, key1, std1, key2, std2, label1, label2 in panels:
        y1 = [r[key1] for r in results]
        s1 = [r[std1] for r in results]
        y2 = [r[key2] for r in results]
        s2 = [r[std2] for r in results]
        lo1, hi1 = _ci95_bounds(y1, s1, coverage, low_clip=0.0)
        lo2, hi2 = _ci95_bounds(y2, s2, coverage, low_clip=0.0)
        ax.plot(x, y1, marker="o", linewidth=2.0, color=colors[0], label=label1)
        ax.plot(x, y2, marker="s", linewidth=2.0, color=colors[1], label=label2)
        ax.fill_between(x, lo1, hi1, color=colors[0], alpha=0.12)
        ax.fill_between(x, lo2, hi2, color=colors[1], alpha=0.12)
        ax.grid(True, alpha=0.35)
        ax.set_ylabel(title)
        ax.legend(frameon=True, loc="best")

    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_ylim(0.0, 1.05)
    axes[-1].set_xlabel("k (neuron index)")
    _set_sparse_xticks(axes[-1], x)
    axes[0].set_title("Synthetic feature recovery diagnostics")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
