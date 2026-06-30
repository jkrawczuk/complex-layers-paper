#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"

MODELS = [
    ("cpl", "cpl"),
    ("svm", "svm"),
    ("logreg", "logreg"),
]

COLON_LOW = {
    "cpl": ROOT / "results/paper_runs_v2/colon_cancer/cpl_n31/balanced/C2",
    "svm": ROOT / "results/paper_runs_v2/colon_cancer/svm_n31/C1p3",
    "logreg": ROOT / "results/paper_runs_v2/colon_cancer/logreg_n31/C3p4",
}

COLON_HIGH = {
    "cpl": ROOT / "results/paper_runs_v2/colon_cancer/cpl_n31/balanced/C8",
    "svm": ROOT / "results/paper_runs_v2/colon_cancer/svm_n31/C5p2",
    "logreg": ROOT / "results/paper_runs_v2/colon_cancer/logreg_n31/C5p2",
}

SYNTH_LOW = {
    "cpl": ROOT / "results/paper_runs_v2/synthetic_signal/cpl_n31/balanced/C3p2",
    "svm": ROOT / "results/paper_runs_v2/synthetic_signal/svm_n31/C2",
    "logreg": ROOT / "results/paper_runs_v2/synthetic_signal/logreg_n31/C4",
}

SYNTH_HIGH = {
    "cpl": ROOT / "results/paper_runs_v2/synthetic_signal/cpl_n31/balanced/C4p6",
    "svm": ROOT / "results/paper_runs_v2/synthetic_signal/svm_n31/C3p4",
    "logreg": ROOT / "results/paper_runs_v2/synthetic_signal/logreg_n31/C10",
}

COLORS = {
    "vote": "#1f77b4",
    "neuron": "#d62728",
    "weighted": "#2ca02c",
    "features": "#9467bd",
    "coverage": "#444444",
    "recall": "#1f77b4",
    "weighted_recall": "#d62728",
    "precision": "#1f77b4",
    "noise_fraction": "#d62728",
    "informative": "#1f77b4",
    "noise": "#d62728",
    "loss": "#2ca02c",
    "l1": "#ff7f0e",
}


PANEL_FONT_SIZE = 17

plt.rcParams.update(
    {
        "font.size": PANEL_FONT_SIZE,
        "axes.titlesize": PANEL_FONT_SIZE,
        "axes.labelsize": PANEL_FONT_SIZE,
        "xtick.labelsize": PANEL_FONT_SIZE,
        "ytick.labelsize": PANEL_FONT_SIZE,
        "legend.fontsize": PANEL_FONT_SIZE,
        "figure.titlesize": PANEL_FONT_SIZE,
    }
)


def _read_csv(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _metrics(bundle: Path):
    return _read_csv(bundle / "metrics" / "results.csv")


def _weighted(bundle: Path):
    path = bundle / "metrics" / "weighted_vote_results.csv"
    return _read_csv(path) if path.exists() else []


def _col(rows, key):
    return np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=float)


def _ci(mean, std, coverage, low=None, high=None):
    half = 1.96 * std / np.sqrt(np.maximum(coverage, 1.0))
    lo = mean - half
    hi = mean + half
    if low is not None:
        lo = np.maximum(lo, low)
    if high is not None:
        hi = np.minimum(hi, high)
    return lo, hi


def _sparse_xticks(ax, x):
    if len(x) <= 8:
        ax.set_xticks(x)
        return
    x_min = int(x.min())
    x_max = int(x.max())
    ticks = [x_min]
    tick = x_min + 5
    while tick < x_max - 2:
        ticks.append(tick)
        tick += 5
    if x_max not in ticks:
        ticks.append(x_max)
    ax.set_xticks(ticks)


def _style_axis(ax):
    ax.grid(True, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_diagnostics_panel(bundles, out_path: Path):
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(17.8, 12.6),
        dpi=180,
        sharey="row",
        gridspec_kw={"height_ratios": [1.25, 1.0, 0.55, 1.0], "hspace": 0.24, "wspace": 0.08},
    )
    accuracy_handles = []
    accuracy_labels = []
    diagnostic_handles = []
    diagnostic_labels = []

    for col, (model, label) in enumerate(MODELS):
        rows = _metrics(bundles[model])
        weighted = _weighted(bundles[model])
        x = _col(rows, "n_neurons")
        coverage = _col(rows, "coverage_n")

        ax_acc = axes[0, col]
        vote = _col(rows, "vote_accuracy")
        vote_std = _col(rows, "vote_accuracy_std")
        neuron = _col(rows, "neuron_accuracy")
        neuron_std = _col(rows, "neuron_accuracy_std")
        vote_lo, vote_hi = _ci(vote, vote_std, coverage, low=0.0, high=1.0)
        neuron_lo, neuron_hi = _ci(neuron, neuron_std, coverage, low=0.0, high=1.0)

        line_neuron = ax_acc.plot(
            x, neuron, marker="s", linewidth=2.2, color=COLORS["neuron"], label="single neuron"
        )[0]
        line_vote = ax_acc.plot(
            x, vote, marker="o", linewidth=2.2, color=COLORS["vote"], label="uniform vote"
        )[0]
        ax_acc.fill_between(x, neuron_lo, neuron_hi, color=COLORS["neuron"], alpha=0.10)
        ax_acc.fill_between(x, vote_lo, vote_hi, color=COLORS["vote"], alpha=0.10)

        line_weighted = None
        if weighted:
            wx = _col(weighted, "n_neurons")
            wcov = _col(weighted, "coverage_n")
            wvote = _col(weighted, "vote_accuracy")
            wstd = _col(weighted, "vote_accuracy_std")
            wlo, whi = _ci(wvote, wstd, wcov, low=0.0, high=1.0)
            line_weighted = ax_acc.plot(
                wx,
                wvote,
                marker="^",
                linestyle="--",
                linewidth=2.2,
                color=COLORS["weighted"],
                label="weighted vote",
            )[0]
            ax_acc.fill_between(wx, wlo, whi, color=COLORS["weighted"], alpha=0.09)

        ax_acc.set_title(label)
        ax_acc.set_ylim(0.0, 1.0)
        ax_acc.tick_params(axis="x", labelbottom=False)
        _style_axis(ax_acc)

        ax_features = axes[1, col]
        features = _col(rows, "features_nth_neuron")
        features_std = _col(rows, "features_nth_neuron_std")
        ax_features.plot(x, features, marker="o", linewidth=2.3, color=COLORS["features"])
        ax_features.fill_between(
            x,
            np.maximum(0.0, features - features_std),
            features + features_std,
            color=COLORS["features"],
            alpha=0.13,
        )
        ax_features.tick_params(axis="x", labelbottom=False)
        _style_axis(ax_features)

        ax_splits = axes[2, col]
        ax_splits.plot(x, coverage, marker="o", linewidth=2.0, color=COLORS["coverage"])
        ax_splits.set_ylim(0, 105)
        ax_splits.tick_params(axis="x", labelbottom=False)
        _style_axis(ax_splits)

        ax_diag = axes[3, col]
        loss = _col(rows, "loss")
        loss_std = _col(rows, "loss_std")
        l1_norm = _col(rows, "l1_norm")
        l1_norm_std = _col(rows, "l1_norm_std")

        loss_scale = np.max(np.abs(loss)) if len(loss) and np.max(np.abs(loss)) != 0 else 1.0
        l1_scale = np.max(np.abs(l1_norm)) if len(l1_norm) and np.max(np.abs(l1_norm)) != 0 else 1.0
        loss_norm = loss / loss_scale
        loss_norm_std = loss_std / loss_scale
        l1_norm_rel = l1_norm / l1_scale
        l1_norm_rel_std = l1_norm_std / l1_scale

        line_loss = ax_diag.plot(
            x, loss_norm, marker="o", linewidth=2.2, color=COLORS["loss"], label="training loss"
        )[0]
        ax_diag.fill_between(
            x,
            np.maximum(0.0, loss_norm - loss_norm_std),
            loss_norm + loss_norm_std,
            color=COLORS["loss"],
            alpha=0.10,
        )
        line_l1 = ax_diag.plot(
            x, l1_norm_rel, marker="s", linewidth=2.2, color=COLORS["l1"], label="L1 norm"
        )[0]
        ax_diag.fill_between(
            x,
            np.maximum(0.0, l1_norm_rel - l1_norm_rel_std),
            l1_norm_rel + l1_norm_rel_std,
            color=COLORS["l1"],
            alpha=0.10,
        )
        ax_diag.axhline(1.0, color="#777777", linewidth=1.0, alpha=0.35)
        ax_diag.set_ylim(0.0, 1.12)
        _sparse_xticks(ax_diag, x)
        _style_axis(ax_diag)

        if col == 0:
            accuracy_handles = [line_neuron, line_vote]
            accuracy_labels = ["single neuron", "uniform vote"]
            if line_weighted is not None:
                accuracy_handles.append(line_weighted)
                accuracy_labels.append("weighted vote")
            diagnostic_handles = [line_loss, line_l1]
            diagnostic_labels = ["training loss", "L1 norm"]
            ax_acc.set_ylabel("accuracy")
            ax_features.set_ylabel("selected features")
            ax_splits.set_ylabel("available\nCV splits")
            ax_diag.set_ylabel("max-scaled\nvalue")

    fig.supxlabel("neuron depth n", x=0.5, y=0.020, fontsize=PANEL_FONT_SIZE)
    fig.legend(
        accuracy_handles,
        accuracy_labels,
        loc="upper center",
        bbox_to_anchor=(0.33, 0.985),
        ncol=3,
        frameon=False,
        fontsize=PANEL_FONT_SIZE,
    )
    fig.legend(
        diagnostic_handles,
        diagnostic_labels,
        loc="lower center",
        bbox_to_anchor=(0.24, 0.003),
        ncol=2,
        frameon=False,
        fontsize=PANEL_FONT_SIZE,
    )
    fig.subplots_adjust(top=0.90, bottom=0.09)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_panel(bundles, out_path: Path):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17.8, 7.0),
        dpi=180,
        sharey="row",
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.10, "wspace": 0.08},
    )
    legend_handles = []
    legend_labels = []

    for col, (model, label) in enumerate(MODELS):
        rows = _metrics(bundles[model])
        weighted = _weighted(bundles[model])
        ax = axes[0, col]
        ax_cov = axes[1, col]

        x = _col(rows, "n_neurons")
        coverage = _col(rows, "coverage_n")
        vote = _col(rows, "vote_accuracy")
        vote_std = _col(rows, "vote_accuracy_std")
        neuron = _col(rows, "neuron_accuracy")
        neuron_std = _col(rows, "neuron_accuracy_std")

        vote_lo, vote_hi = _ci(vote, vote_std, coverage, low=0.0, high=1.0)
        neuron_lo, neuron_hi = _ci(neuron, neuron_std, coverage, low=0.0, high=1.0)

        line_vote = ax.plot(x, vote, marker="o", linewidth=2.3, color=COLORS["vote"], label="uniform vote")[0]
        line_neuron = ax.plot(
            x,
            neuron,
            marker="s",
            linewidth=2.3,
            color=COLORS["neuron"],
            label="single neuron",
        )[0]
        ax.fill_between(x, vote_lo, vote_hi, color=COLORS["vote"], alpha=0.10)
        ax.fill_between(x, neuron_lo, neuron_hi, color=COLORS["neuron"], alpha=0.10)

        if weighted:
            wx = _col(weighted, "n_neurons")
            wcov = _col(weighted, "coverage_n")
            wvote = _col(weighted, "vote_accuracy")
            wstd = _col(weighted, "vote_accuracy_std")
            wlo, whi = _ci(wvote, wstd, wcov, low=0.0, high=1.0)
            line_weighted = ax.plot(
                wx,
                wvote,
                marker="^",
                linestyle="--",
                linewidth=2.3,
                color=COLORS["weighted"],
                label="weighted vote",
            )[0]
            ax.fill_between(wx, wlo, whi, color=COLORS["weighted"], alpha=0.09)
        else:
            line_weighted = None

        ax_cov.plot(x, coverage, marker="o", linewidth=2.0, color=COLORS["coverage"])
        ax.set_title(label)
        ax.set_ylim(0.5, 1.0)
        ax_cov.set_ylim(0, 105)
        ax.tick_params(axis="x", labelbottom=False)
        _sparse_xticks(ax_cov, x)
        _style_axis(ax)
        _style_axis(ax_cov)

        if col == 0:
            legend_handles = [line_neuron, line_vote]
            legend_labels = ["single neuron", "uniform vote"]
            if line_weighted is not None:
                legend_handles.append(line_weighted)
                legend_labels.append("weighted vote")
            ax.set_ylabel("accuracy")
            ax_cov.set_ylabel("available\nCV splits")

    fig.supxlabel("neuron depth n", y=0.01, fontsize=PANEL_FONT_SIZE)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=True,
        fontsize=PANEL_FONT_SIZE,
    )
    fig.subplots_adjust(top=0.80, bottom=0.13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_features_panel(bundles, out_path: Path, title_prefix: str):
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 4.8), dpi=180, sharey=True)
    for col, (model, label) in enumerate(MODELS):
        rows = _metrics(bundles[model])
        ax = axes[col]
        x = _col(rows, "n_neurons")
        features = _col(rows, "features_nth_neuron")
        features_std = _col(rows, "features_nth_neuron_std")
        ax.plot(x, features, marker="o", linewidth=2.4, color=COLORS["features"])
        ax.fill_between(
            x,
            np.maximum(0.0, features - features_std),
            features + features_std,
            color=COLORS["features"],
            alpha=0.13,
        )
        ax.set_title(label)
        _sparse_xticks(ax, x)
        _style_axis(ax)
        if col == 0:
            ax.set_ylabel("selected features")
    fig.supxlabel("neuron depth n", y=0.02, fontsize=PANEL_FONT_SIZE)
    fig.subplots_adjust(bottom=0.20, top=0.88, wspace=0.08)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_recovery_panel(bundles, out_path: Path):
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(17.8, 10.2),
        dpi=180,
        sharey="row",
        gridspec_kw={"hspace": 0.18, "wspace": 0.10},
    )

    for col, (model, label) in enumerate(MODELS):
        rows = _metrics(bundles[model])
        x = _col(rows, "n_neurons")

        panels = [
            (
                axes[0, col],
                [
                    ("cumulative_feature_recall", "informative features", COLORS["informative"], "o"),
                ],
                "cumulative recall",
                (0.0, 1.05),
            ),
            (
                axes[1, col],
                [
                    ("feature_precision_nth", "informative features", COLORS["informative"], "o"),
                    ("noise_fraction_nth", "noise features", COLORS["noise"], "s"),
                ],
                "per-neuron fraction",
                (0.0, 1.05),
            ),
            (
                axes[2, col],
                [
                    ("informative_selected_nth", "informative features", COLORS["informative"], "o"),
                    ("noise_selected_nth", "noise features", COLORS["noise"], "s"),
                ],
                "selected features",
                None,
            ),
        ]

        for row_idx, (ax, series, ylabel, ylim) in enumerate(panels):
            for key, label_text, color, marker in series:
                y = _col(rows, key)
                std = _col(rows, f"{key}_std")
                ax.plot(x, y, marker=marker, linewidth=2.2, color=color, label=label_text)
                ax.fill_between(x, np.maximum(0.0, y - std), y + std, color=color, alpha=0.10)
            if ylim:
                ax.set_ylim(*ylim)
            if row_idx == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 2:
                _sparse_xticks(ax, x)
            _style_axis(ax)

    fig.supxlabel("neuron depth n", y=0.005, fontsize=PANEL_FONT_SIZE)
    legend_handles = [
        Line2D([0], [0], color=COLORS["informative"], marker="o", linewidth=2.2, label="informative features"),
        Line2D([0], [0], color=COLORS["noise"], marker="s", linewidth=2.2, label="noise features"),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.32, 0.965),
        ncol=2,
        frameon=False,
        fontsize=PANEL_FONT_SIZE,
    )
    fig.subplots_adjust(top=0.86, bottom=0.10)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_diagnostics_panel(COLON_LOW, OUT_DIR / "colon_diagnostics_low.png")
    save_diagnostics_panel(COLON_HIGH, OUT_DIR / "colon_diagnostics_high.png")
    save_diagnostics_panel(SYNTH_LOW, OUT_DIR / "synthetic_diagnostics_low.png")
    save_diagnostics_panel(SYNTH_HIGH, OUT_DIR / "synthetic_diagnostics_high.png")
    save_recovery_panel(SYNTH_LOW, OUT_DIR / "synthetic_recovery_low.png")
    save_recovery_panel(SYNTH_HIGH, OUT_DIR / "synthetic_recovery_high.png")


if __name__ == "__main__":
    main()
