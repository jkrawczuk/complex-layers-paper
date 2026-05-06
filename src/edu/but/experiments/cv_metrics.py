import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from edu.but.experiments.cv_io import (
    write_per_split_metrics_csv,
    write_results_csv,
)
from edu.but.experiments.cv_worker import classification_metrics


_NEURON_SUMMARY_KEYS = [
    "features_nth_neuron",
    "informative_selected_nth",
    "noise_selected_nth",
    "feature_precision_nth",
    "feature_recall_nth",
    "feature_weight_recall_nth",
    "noise_fraction_nth",
    "cumulative_informative_selected",
    "cumulative_feature_precision",
    "cumulative_feature_recall",
    "cumulative_feature_weight_recall",
    "cumulative_noise_fraction",
    "loss",
    "l1_norm",
]


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    if value in (None, ""):
        return None
    return float(value)


def _to_int(value):
    if value in (None, ""):
        return None
    return int(value)


def _mean_std(values):
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    mean_v = float(np.mean(clean))
    std_v = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    return mean_v, std_v


def _split_key(row):
    return int(row["split_id"]), int(row["repeat_id"]), int(row["fold_id"])


def _metadata_from_rows(rows):
    first = rows[0]
    return {
        "dataset": first["dataset"],
        "omics": first["omics"],
        "model": first["model"],
        "C": first["C"],
        "class_weight": first.get("class_weight", ""),
    }


def _metadata_from_run_stats(bundle_dir: Path, fallback_rows):
    stats_path = bundle_dir / "run_stats.json"
    if not stats_path.exists():
        return _metadata_from_rows(fallback_rows)
    with stats_path.open(encoding="utf-8") as f:
        stats = json.load(f)
    return {
        "dataset": stats.get("dataset", ""),
        "omics": stats.get("omics", "all"),
        "model": stats.get("base_model", ""),
        "C": stats.get("C", ""),
        "class_weight": stats.get("class_weight", ""),
    }


def _pivot_predictions(raw_rows):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    truth = defaultdict(lambda: defaultdict(dict))
    for row in raw_rows:
        key = _split_key(row)
        set_name = row["set"]
        sample_index = int(row["sample_index"])
        truth[key][set_name][sample_index] = row["y_true"]
        for col, value in row.items():
            if not col.startswith("pred_n") or value in (None, ""):
                continue
            n_neuron = int(col.removeprefix("pred_n"))
            grouped[key][set_name][n_neuron][sample_index] = value
    return grouped, truth


def _load_neuron_summary(rows):
    out = {}
    for row in rows:
        key = _split_key(row)
        n = int(row["n_neuron"])
        parsed = {k: _to_float(row.get(k)) for k in _NEURON_SUMMARY_KEYS}
        out[(key, n)] = parsed
    return out


def _load_class_labels(bundle_dir: Path):
    path = bundle_dir / "class_labels.json"
    if not path.exists():
        return {"0": "0", "1": "1"}
    with path.open(encoding="utf-8") as f:
        labels = json.load(f)
    return {"0": str(labels.get("0", "0")), "1": str(labels.get("1", "1"))}


def _class_counts_text(values, labels):
    counts = {label: 0 for label in labels.values()}
    for value in values:
        counts[labels[str(value)]] = counts.get(labels[str(value)], 0) + 1
    return ";".join(f"{label}:{counts[label]}" for label in sorted(counts))


def _minority_class(values, labels):
    counts = {encoded: 0 for encoded in labels}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    minority_encoded = min(counts, key=lambda k: (counts[k], k))
    return minority_encoded, labels[minority_encoded]


def _uniform_vote(preds_by_n, sample_indices, n_neurons):
    out = []
    for sample_index in sample_indices:
        labels = [preds_by_n[n][sample_index] for n in range(1, n_neurons + 1)]
        counts = defaultdict(float)
        for label in labels:
            counts[label] += 1.0
        max_count = max(counts.values())
        winners = {label for label, count in counts.items() if count == max_count}
        out.append(labels[0] if len(winners) > 1 else next(iter(winners)))
    return out


def _weighted_vote(preds_by_n, sample_indices, weights):
    out = []
    n_neurons = len(weights)
    for sample_index in sample_indices:
        labels = [preds_by_n[n][sample_index] for n in range(1, n_neurons + 1)]
        scores = defaultdict(float)
        for label, weight in zip(labels, weights):
            scores[label] += weight
        max_score = max(scores.values())
        winners = {label for label, score in scores.items() if score == max_score}
        out.append(labels[0] if len(winners) > 1 else next(iter(winners)))
    return out


def _quality_weights(train_balanced_by_n, n_neurons):
    raw = [max(0.0, float(train_balanced_by_n.get(n, 0.0)) - 0.5) for n in range(1, n_neurons + 1)]
    total = float(sum(raw))
    if total <= 0.0:
        return [1.0 / n_neurons for _ in range(n_neurons)]
    return [v / total for v in raw]


def _aggregate_per_split_rows(rows):
    by_n = defaultdict(list)
    for row in rows:
        by_n[int(row["n_neurons"])].append(row)

    aggregate = []
    for n in sorted(by_n):
        n_rows = by_n[n]
        out = {"n_neurons": n, "coverage_n": len(n_rows)}
        mapping = {
            "mean_accuracy": "vote_accuracy",
            "std_accuracy": "vote_accuracy",
            "mean_balanced_accuracy": "vote_balanced_accuracy",
            "std_balanced_accuracy": "vote_balanced_accuracy",
            "mean_minority_precision": "vote_minority_precision",
            "std_minority_precision": "vote_minority_precision",
            "mean_minority_recall": "vote_minority_recall",
            "std_minority_recall": "vote_minority_recall",
            "mean_minority_f1": "vote_minority_f1",
            "std_minority_f1": "vote_minority_f1",
            "mean_train_accuracy": "vote_train_accuracy",
            "std_train_accuracy": "vote_train_accuracy",
            "mean_train_balanced_accuracy": "vote_train_balanced_accuracy",
            "std_train_balanced_accuracy": "vote_train_balanced_accuracy",
            "mean_train_minority_precision": "vote_train_minority_precision",
            "std_train_minority_precision": "vote_train_minority_precision",
            "mean_train_minority_recall": "vote_train_minority_recall",
            "std_train_minority_recall": "vote_train_minority_recall",
            "mean_train_minority_f1": "vote_train_minority_f1",
            "std_train_minority_f1": "vote_train_minority_f1",
            "features": "features_nth_neuron",
            "features_std": "features_nth_neuron",
            "neuron_accuracy": "neuron_accuracy",
            "neuron_accuracy_std": "neuron_accuracy",
            "neuron_balanced_accuracy": "neuron_balanced_accuracy",
            "neuron_balanced_accuracy_std": "neuron_balanced_accuracy",
            "neuron_minority_precision": "neuron_minority_precision",
            "neuron_minority_precision_std": "neuron_minority_precision",
            "neuron_minority_recall": "neuron_minority_recall",
            "neuron_minority_recall_std": "neuron_minority_recall",
            "neuron_minority_f1": "neuron_minority_f1",
            "neuron_minority_f1_std": "neuron_minority_f1",
            "neuron_train_accuracy": "neuron_train_accuracy",
            "neuron_train_accuracy_std": "neuron_train_accuracy",
            "neuron_train_balanced_accuracy": "neuron_train_balanced_accuracy",
            "neuron_train_balanced_accuracy_std": "neuron_train_balanced_accuracy",
            "neuron_train_minority_precision": "neuron_train_minority_precision",
            "neuron_train_minority_precision_std": "neuron_train_minority_precision",
            "neuron_train_minority_recall": "neuron_train_minority_recall",
            "neuron_train_minority_recall_std": "neuron_train_minority_recall",
            "neuron_train_minority_f1": "neuron_train_minority_f1",
            "neuron_train_minority_f1_std": "neuron_train_minority_f1",
            "loss": "loss",
            "loss_std": "loss",
            "l1_norm": "l1_norm",
            "l1_norm_std": "l1_norm",
        }
        for out_key, in_key in mapping.items():
            mean_v, std_v = _mean_std(_to_float(r.get(in_key)) for r in n_rows)
            out[out_key] = std_v if out_key.startswith("std") or out_key.endswith("_std") else mean_v

        for key in [
            "informative_selected_nth",
            "noise_selected_nth",
            "feature_precision_nth",
            "feature_recall_nth",
            "feature_weight_recall_nth",
            "noise_fraction_nth",
            "cumulative_informative_selected",
            "cumulative_feature_precision",
            "cumulative_feature_recall",
            "cumulative_feature_weight_recall",
            "cumulative_noise_fraction",
        ]:
            mean_v, std_v = _mean_std(_to_float(r.get(key)) for r in n_rows)
            out[key] = mean_v
            out[f"{key}_std"] = std_v
        aggregate.append(out)
    return aggregate


def _write_weighted_per_split(path: Path, rows):
    fieldnames = [
        "dataset",
        "omics",
        "model",
        "C",
        "class_weight",
        "weight_metric",
        "split_id",
        "repeat_id",
        "fold_id",
        "n_neurons",
        "minority_class",
        "weights",
        "vote_accuracy",
        "vote_balanced_accuracy",
        "vote_minority_precision",
        "vote_minority_recall",
        "vote_minority_f1",
        "vote_train_accuracy",
        "vote_train_balanced_accuracy",
        "vote_train_minority_precision",
        "vote_train_minority_recall",
        "vote_train_minority_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_weighted_results(path: Path, rows):
    fieldnames = [
        "weight_metric",
        "n_neurons",
        "coverage_n",
        "vote_accuracy",
        "vote_accuracy_std",
        "vote_balanced_accuracy",
        "vote_balanced_accuracy_std",
        "vote_minority_precision",
        "vote_minority_precision_std",
        "vote_minority_recall",
        "vote_minority_recall_std",
        "vote_minority_f1",
        "vote_minority_f1_std",
        "vote_train_accuracy",
        "vote_train_accuracy_std",
        "vote_train_balanced_accuracy",
        "vote_train_balanced_accuracy_std",
        "vote_train_minority_precision",
        "vote_train_minority_precision_std",
        "vote_train_minority_recall",
        "vote_train_minority_recall_std",
        "vote_train_minority_f1",
        "vote_train_minority_f1_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_weighted_rows(rows):
    out_rows = []
    by_strategy_n = defaultdict(list)
    for row in rows:
        by_strategy_n[(row["weight_metric"], int(row["n_neurons"]))].append(row)

    for (weight_metric, n), n_rows in sorted(by_strategy_n.items()):
        out = {
            "weight_metric": weight_metric,
            "n_neurons": n,
            "coverage_n": len(n_rows),
        }
        for key in [
            "vote_accuracy",
            "vote_balanced_accuracy",
            "vote_minority_precision",
            "vote_minority_recall",
            "vote_minority_f1",
            "vote_train_accuracy",
            "vote_train_balanced_accuracy",
            "vote_train_minority_precision",
            "vote_train_minority_recall",
            "vote_train_minority_f1",
        ]:
            mean_v, std_v = _mean_std(_to_float(r.get(key)) for r in n_rows)
            out[key] = mean_v
            out[f"{key}_std"] = std_v
        out_rows.append(out)
    return out_rows


def compute_cv_metrics(bundle_dir: Path):
    bundle_dir = Path(bundle_dir)
    raw_rows = _read_csv(bundle_dir / "raw_predictions.csv")
    neuron_rows = _read_csv(bundle_dir / "neuron_summary.csv")
    if not raw_rows:
        raise ValueError(f"No raw predictions in {bundle_dir}")

    meta = _metadata_from_run_stats(bundle_dir, raw_rows)
    class_labels = _load_class_labels(bundle_dir)
    predictions, truth = _pivot_predictions(raw_rows)
    neuron_summary = _load_neuron_summary(neuron_rows)

    per_split_rows = []
    weighted_rows = []
    for key in sorted(predictions):
        train_samples = sorted(truth[key]["train"])
        test_samples = sorted(truth[key]["test"])
        y_train = [truth[key]["train"][idx] for idx in train_samples]
        y_test = [truth[key]["test"][idx] for idx in test_samples]
        built_n = max(predictions[key]["train"].keys())
        minority_encoded, minority_label = _minority_class(y_train, class_labels)
        class_counts_train = _class_counts_text(y_train, class_labels)
        class_counts_test = _class_counts_text(y_test, class_labels)

        train_balanced_by_n = {}
        neuron_train_metrics_by_n = {}
        neuron_test_metrics_by_n = {}
        for n in range(1, built_n + 1):
            train_pred = [predictions[key]["train"][n][idx] for idx in train_samples]
            test_pred = [predictions[key]["test"][n][idx] for idx in test_samples]
            neuron_train_metrics = classification_metrics(y_train, train_pred, minority_encoded)
            neuron_test_metrics = classification_metrics(y_test, test_pred, minority_encoded)
            neuron_train_metrics_by_n[n] = neuron_train_metrics
            neuron_test_metrics_by_n[n] = neuron_test_metrics
            train_balanced_by_n[n] = neuron_train_metrics["balanced_accuracy"]

        for n in range(1, built_n + 1):
            train_uniform = _uniform_vote(predictions[key]["train"], train_samples, n)
            test_uniform = _uniform_vote(predictions[key]["test"], test_samples, n)
            train_vote_metrics = classification_metrics(y_train, train_uniform, minority_encoded)
            test_vote_metrics = classification_metrics(y_test, test_uniform, minority_encoded)
            neuron_train_metrics = neuron_train_metrics_by_n[n]
            neuron_test_metrics = neuron_test_metrics_by_n[n]
            summary_n = neuron_summary[(key, n)]

            per_split_rows.append(
                {
                    **meta,
                    "split_id": key[0],
                    "repeat_id": key[1],
                    "fold_id": key[2],
                    "n_neurons_requested": meta.get("n_neurons_requested", built_n),
                    "n_neurons_built_in_split": built_n,
                    "n_neurons": n,
                    "minority_class": minority_label,
                    "vote_accuracy": test_vote_metrics["accuracy"],
                    "vote_balanced_accuracy": test_vote_metrics["balanced_accuracy"],
                    "vote_minority_precision": test_vote_metrics["minority_precision"],
                    "vote_minority_recall": test_vote_metrics["minority_recall"],
                    "vote_minority_f1": test_vote_metrics["minority_f1"],
                    "vote_train_accuracy": train_vote_metrics["accuracy"],
                    "vote_train_balanced_accuracy": train_vote_metrics["balanced_accuracy"],
                    "vote_train_minority_precision": train_vote_metrics["minority_precision"],
                    "vote_train_minority_recall": train_vote_metrics["minority_recall"],
                    "vote_train_minority_f1": train_vote_metrics["minority_f1"],
                    "neuron_accuracy": neuron_test_metrics["accuracy"],
                    "neuron_balanced_accuracy": neuron_test_metrics["balanced_accuracy"],
                    "neuron_minority_precision": neuron_test_metrics["minority_precision"],
                    "neuron_minority_recall": neuron_test_metrics["minority_recall"],
                    "neuron_minority_f1": neuron_test_metrics["minority_f1"],
                    "neuron_train_accuracy": neuron_train_metrics["accuracy"],
                    "neuron_train_balanced_accuracy": neuron_train_metrics["balanced_accuracy"],
                    "neuron_train_minority_precision": neuron_train_metrics["minority_precision"],
                    "neuron_train_minority_recall": neuron_train_metrics["minority_recall"],
                    "neuron_train_minority_f1": neuron_train_metrics["minority_f1"],
                    **summary_n,
                    "y_train_size": len(y_train),
                    "y_test_size": len(y_test),
                    "class_counts_train": class_counts_train,
                    "class_counts_test": class_counts_test,
                    "class_label_0": class_labels.get("0", ""),
                    "class_label_1": class_labels.get("1", ""),
                }
            )

            weights = _quality_weights(train_balanced_by_n, n)
            train_weighted = _weighted_vote(predictions[key]["train"], train_samples, weights)
            test_weighted = _weighted_vote(predictions[key]["test"], test_samples, weights)
            train_weighted_metrics = classification_metrics(y_train, train_weighted, minority_encoded)
            test_weighted_metrics = classification_metrics(y_test, test_weighted, minority_encoded)
            weighted_rows.append(
                {
                    **meta,
                    "weight_metric": "train_balanced_accuracy_minus_chance",
                    "split_id": key[0],
                    "repeat_id": key[1],
                    "fold_id": key[2],
                    "n_neurons": n,
                    "minority_class": minority_label,
                    "weights": ";".join(f"{w:.8g}" for w in weights),
                    "vote_accuracy": test_weighted_metrics["accuracy"],
                    "vote_balanced_accuracy": test_weighted_metrics["balanced_accuracy"],
                    "vote_minority_precision": test_weighted_metrics["minority_precision"],
                    "vote_minority_recall": test_weighted_metrics["minority_recall"],
                    "vote_minority_f1": test_weighted_metrics["minority_f1"],
                    "vote_train_accuracy": train_weighted_metrics["accuracy"],
                    "vote_train_balanced_accuracy": train_weighted_metrics["balanced_accuracy"],
                    "vote_train_minority_precision": train_weighted_metrics["minority_precision"],
                    "vote_train_minority_recall": train_weighted_metrics["minority_recall"],
                    "vote_train_minority_f1": train_weighted_metrics["minority_f1"],
                }
            )

    metrics_dir = bundle_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_per_split_metrics_csv(metrics_dir / "per_split_metrics.csv", per_split_rows)
    write_results_csv(metrics_dir / "results.csv", _aggregate_per_split_rows(per_split_rows))
    _write_weighted_per_split(metrics_dir / "weighted_vote_per_split_metrics.csv", weighted_rows)
    _write_weighted_results(
        metrics_dir / "weighted_vote_results.csv",
        _aggregate_weighted_rows(weighted_rows),
    )

    with (metrics_dir / "metrics_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_bundle_dir": str(bundle_dir),
                "weighting": "q_i = max(0, train_neuron_balanced_accuracy_i - 0.5)",
                "weighted_vote_tie_break": "first neuron prediction",
            },
            f,
            indent=2,
        )
    print(f"saved metrics dir: {metrics_dir}")
    return metrics_dir
