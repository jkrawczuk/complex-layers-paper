import csv
import json
from pathlib import Path


def resolve_output_bundle(out_csv: Path):
    if out_csv.suffix.lower() == ".csv":
        out_dir = out_csv.with_suffix("")
        csv_path = out_dir / out_csv.name
    else:
        out_dir = out_csv
        csv_path = out_dir / "results.csv"
    return out_dir, csv_path


def format_c_value(c_value: float) -> str:
    return f"{c_value:g}".replace(".", "p")


def default_output_bundle_path(
    data_path: Path,
    omics: str,
    base_model: str,
    c_value: float,
    n_neurons: int,
    class_weight: str = "balanced",
) -> Path:
    dataset_name = data_path.stem
    omics_name = "all" if not omics else omics.replace(",", "-")
    c_name = f"C{format_c_value(c_value)}"
    model_name = f"{base_model}_n{n_neurons}"
    base = Path("results") / dataset_name
    if omics_name != "all":
        base = base / omics_name
    if base_model == "cpl_lp":
        return base / model_name / class_weight / c_name
    return base / model_name / c_name


def write_results_csv(csv_path: Path, results):
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = [
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
            "features_nth_neuron",
            "features_nth_neuron_std",
            "informative_selected_nth",
            "informative_selected_nth_std",
            "noise_selected_nth",
            "noise_selected_nth_std",
            "feature_precision_nth",
            "feature_precision_nth_std",
            "feature_recall_nth",
            "feature_recall_nth_std",
            "feature_weight_recall_nth",
            "feature_weight_recall_nth_std",
            "noise_fraction_nth",
            "noise_fraction_nth_std",
            "cumulative_informative_selected",
            "cumulative_informative_selected_std",
            "cumulative_feature_precision",
            "cumulative_feature_precision_std",
            "cumulative_feature_recall",
            "cumulative_feature_recall_std",
            "cumulative_feature_weight_recall",
            "cumulative_feature_weight_recall_std",
            "cumulative_noise_fraction",
            "cumulative_noise_fraction_std",
            "neuron_accuracy",
            "neuron_accuracy_std",
            "neuron_balanced_accuracy",
            "neuron_balanced_accuracy_std",
            "neuron_minority_precision",
            "neuron_minority_precision_std",
            "neuron_minority_recall",
            "neuron_minority_recall_std",
            "neuron_minority_f1",
            "neuron_minority_f1_std",
            "neuron_train_accuracy",
            "neuron_train_accuracy_std",
            "neuron_train_balanced_accuracy",
            "neuron_train_balanced_accuracy_std",
            "neuron_train_minority_precision",
            "neuron_train_minority_precision_std",
            "neuron_train_minority_recall",
            "neuron_train_minority_recall_std",
            "neuron_train_minority_f1",
            "neuron_train_minority_f1_std",
            "loss",
            "loss_std",
            "l1_norm",
            "l1_norm_std",
        ]
        writer.writerow(header)
        for row in results:
            writer.writerow(
                [
                    row["n_neurons"],
                    row["coverage_n"],
                    row["mean_accuracy"],
                    row["std_accuracy"],
                    row["mean_balanced_accuracy"],
                    row["std_balanced_accuracy"],
                    row["mean_minority_precision"],
                    row["std_minority_precision"],
                    row["mean_minority_recall"],
                    row["std_minority_recall"],
                    row["mean_minority_f1"],
                    row["std_minority_f1"],
                    row["mean_train_accuracy"],
                    row["std_train_accuracy"],
                    row["mean_train_balanced_accuracy"],
                    row["std_train_balanced_accuracy"],
                    row["mean_train_minority_precision"],
                    row["std_train_minority_precision"],
                    row["mean_train_minority_recall"],
                    row["std_train_minority_recall"],
                    row["mean_train_minority_f1"],
                    row["std_train_minority_f1"],
                    row["features"],
                    row["features_std"],
                    row.get("informative_selected_nth"),
                    row.get("informative_selected_nth_std"),
                    row.get("noise_selected_nth"),
                    row.get("noise_selected_nth_std"),
                    row.get("feature_precision_nth"),
                    row.get("feature_precision_nth_std"),
                    row.get("feature_recall_nth"),
                    row.get("feature_recall_nth_std"),
                    row.get("feature_weight_recall_nth"),
                    row.get("feature_weight_recall_nth_std"),
                    row.get("noise_fraction_nth"),
                    row.get("noise_fraction_nth_std"),
                    row.get("cumulative_informative_selected"),
                    row.get("cumulative_informative_selected_std"),
                    row.get("cumulative_feature_precision"),
                    row.get("cumulative_feature_precision_std"),
                    row.get("cumulative_feature_recall"),
                    row.get("cumulative_feature_recall_std"),
                    row.get("cumulative_feature_weight_recall"),
                    row.get("cumulative_feature_weight_recall_std"),
                    row.get("cumulative_noise_fraction"),
                    row.get("cumulative_noise_fraction_std"),
                    row["neuron_accuracy"],
                    row["neuron_accuracy_std"],
                    row["neuron_balanced_accuracy"],
                    row["neuron_balanced_accuracy_std"],
                    row["neuron_minority_precision"],
                    row["neuron_minority_precision_std"],
                    row["neuron_minority_recall"],
                    row["neuron_minority_recall_std"],
                    row["neuron_minority_f1"],
                    row["neuron_minority_f1_std"],
                    row["neuron_train_accuracy"],
                    row["neuron_train_accuracy_std"],
                    row["neuron_train_balanced_accuracy"],
                    row["neuron_train_balanced_accuracy_std"],
                    row["neuron_train_minority_precision"],
                    row["neuron_train_minority_precision_std"],
                    row["neuron_train_minority_recall"],
                    row["neuron_train_minority_recall_std"],
                    row["neuron_train_minority_f1"],
                    row["neuron_train_minority_f1_std"],
                    row["loss"],
                    row["loss_std"],
                    row["l1_norm"],
                    row["l1_norm_std"],
                ]
            )


def write_per_split_metrics_csv(csv_path: Path, rows):
    fieldnames = [
        "dataset",
        "omics",
        "model",
        "C",
        "class_weight",
        "split_id",
        "repeat_id",
        "fold_id",
        "n_neurons_requested",
        "n_neurons_built_in_split",
        "n_neurons",
        "minority_class",
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
        "neuron_accuracy",
        "neuron_balanced_accuracy",
        "neuron_minority_precision",
        "neuron_minority_recall",
        "neuron_minority_f1",
        "neuron_train_accuracy",
        "neuron_train_balanced_accuracy",
        "neuron_train_minority_precision",
        "neuron_train_minority_recall",
        "neuron_train_minority_f1",
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
        "y_train_size",
        "y_test_size",
        "class_counts_train",
        "class_counts_test",
        "class_label_0",
        "class_label_1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_per_split_summary_csv(csv_path: Path, rows):
    fieldnames = [
        "dataset",
        "omics",
        "model",
        "C",
        "class_weight",
        "split_id",
        "repeat_id",
        "fold_id",
        "n_neurons_requested",
        "n_neurons_built_in_split",
        "fit_time_sec",
        "minority_class",
        "y_train_size",
        "y_test_size",
        "class_counts_train",
        "class_counts_test",
        "class_label_0",
        "class_label_1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_raw_predictions_csv(csv_path: Path, rows):
    max_n = 0
    for row in rows:
        for key in row:
            if key.startswith("pred_n"):
                max_n = max(max_n, int(key.removeprefix("pred_n")))
    fieldnames = [
        "split_id",
        "repeat_id",
        "fold_id",
        "set",
        "sample_index",
        "y_true",
    ]
    fieldnames.extend(f"pred_n{n}" for n in range(1, max_n + 1))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_neuron_summary_csv(csv_path: Path, rows):
    fieldnames = [
        "split_id",
        "repeat_id",
        "fold_id",
        "n_neuron",
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
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_selected_features_csv(csv_path: Path, rows):
    fieldnames = [
        "split_id",
        "repeat_id",
        "fold_id",
        "n_neuron",
        "feature_index",
        "coefficient",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_run_stats_json(out_dir: Path, stats: dict):
    stats_path = out_dir / "run_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats_path


def update_run_stats_summary(dataset_name: str):
    dataset_dir = Path("results") / dataset_name
    stats_files = sorted(dataset_dir.rglob("run_stats.json"))
    rows = []
    for p in stats_files:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "dataset": data.get("dataset", dataset_name),
                "omic": data.get("omics", "all"),
                "model": data.get("base_model", ""),
                "C": data.get("C", ""),
                "n_neurons_requested": data.get("n_neurons_requested", ""),
                "n_neurons_built": data.get("n_neurons_built", ""),
                "cv_folds": data.get("cv_folds", ""),
                "cv_repeats": data.get("cv_repeats", ""),
                "cv_total_folds": data.get("cv_total_folds", ""),
                "jobs": data.get("jobs", 1),
                "wall_time_sec": data.get("wall_time_sec", ""),
                "max_rss_gb": data.get("max_rss_gb", ""),
                "csv_path": data.get("csv_path", ""),
                "run_dir": data.get("run_dir", str(p.parent)),
            }
        )

    out_csv = dataset_dir / "tables" / "run_stats_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "omic",
                "model",
                "C",
                "n_neurons_requested",
                "n_neurons_built",
                "cv_folds",
                "cv_repeats",
                "cv_total_folds",
                "jobs",
                "wall_time_sec",
                "max_rss_gb",
                "csv_path",
                "run_dir",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_csv
