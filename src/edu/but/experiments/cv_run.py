import concurrent.futures
import json
import time
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from tqdm import tqdm

from edu.but.experiments.cv_io import (
    default_output_bundle_path,
    save_run_stats_json,
    update_run_stats_summary,
    write_neuron_summary_csv,
    write_per_split_summary_csv,
    write_raw_predictions_csv,
    write_selected_features_csv,
)
from edu.but.experiments.cv_monitor import max_rss_bytes_from_rusage
from edu.but.experiments.cv_worker import init_worker, run_single_split
from edu.but.experiments.ground_truth import load_ground_truth_metadata
from edu.but.layers.utils import load_data


def build_splitter(cv_folds: int, cv_repeats: int, random_state: int):
    if cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2 for cross-validation.")
    if cv_repeats < 1:
        raise ValueError("--cv-repeats must be >= 1.")
    if cv_repeats == 1:
        return (
            StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            cv_folds,
        )
    return (
        RepeatedStratifiedKFold(
            n_splits=cv_folds,
            n_repeats=cv_repeats,
            random_state=random_state,
        ),
        cv_folds * cv_repeats,
    )


def _with_split_metadata(row, args, data_path: Path):
    return {
        "dataset": data_path.stem,
        "omics": args.omics or "all",
        "model": args.base_model,
        "C": args.C,
        "class_weight": args.class_weight,
        **row,
    }


def run_cv_predictions(args):
    run_start = time.perf_counter()
    data_path = Path(args.data)
    X, y = load_data(
        data_path,
        label_first=args.label_first,
        has_header=args.has_header,
        omics=args.omics,
    )
    ground_truth = load_ground_truth_metadata(data_path)
    if ground_truth is not None:
        print(f"Loaded ground-truth metadata: {ground_truth['_path']}")

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1.")

    splitter, total_folds = build_splitter(args.cv_folds, args.cv_repeats, args.random_state)
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_output_bundle_path(
        data_path=data_path,
        omics=args.omics,
        base_model=args.base_model,
        c_value=args.C,
        n_neurons=args.n_neurons,
        class_weight=args.class_weight,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running raw CV predictions with {args.cv_folds} folds x {args.cv_repeats} repeats "
        f"(total {total_folds}) and C={args.C} (jobs={args.jobs})"
    )

    split_tasks = []
    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        split_tasks.append(
            {
                "split_idx": split_idx,
                "repeat_idx": ((split_idx - 1) // args.cv_folds) + 1,
                "fold_idx": ((split_idx - 1) % args.cv_folds) + 1,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "cfg": {
                    "n_neurons": args.n_neurons,
                    "base_model": args.base_model,
                    "C": args.C,
                    "class_weight_mode": args.class_weight,
                    "random_state": args.random_state,
                    "verbose": args.verbose,
                    "ground_truth": ground_truth,
                },
            }
        )

    split_results = []
    progress = tqdm(total=total_folds, desc="CV folds", unit="fold")
    init_worker(X, y)

    def _consume(res):
        split_results.append(res)
        progress.update(1)
        progress.set_postfix_str(
            f"repeat={res['repeat_id']}/{args.cv_repeats}, fold={res['fold_id']}/{args.cv_folds}"
        )

    if args.jobs == 1:
        for task in split_tasks:
            _consume(run_single_split(task))
    else:
        init_worker(X, y)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(run_single_split, task) for task in split_tasks]
            for fut in concurrent.futures.as_completed(futures):
                _consume(fut.result())
    progress.close()

    raw_prediction_rows = []
    neuron_summary_rows = []
    selected_feature_rows = []
    split_summaries = []
    max_built = 0
    for res in split_results:
        max_built = max(max_built, int(res["built_n"]))
        split_summaries.append(res["split_summary"])
        raw_prediction_rows.extend(
            row for row in res["raw_prediction_rows"]
        )
        neuron_summary_rows.extend(
            row for row in res["neuron_summary_rows"]
        )
        selected_feature_rows.extend(
            row for row in res["selected_feature_rows"]
        )

    write_raw_predictions_csv(out_dir / "raw_predictions.csv", raw_prediction_rows)
    write_neuron_summary_csv(out_dir / "neuron_summary.csv", neuron_summary_rows)
    write_selected_features_csv(out_dir / "selected_features.csv", selected_feature_rows)
    class_labels = {}
    if split_summaries:
        class_labels = {
            "0": split_summaries[0].get("class_label_0"),
            "1": split_summaries[0].get("class_label_1"),
        }
    with (out_dir / "class_labels.json").open("w", encoding="utf-8") as f:
        json.dump(class_labels, f, indent=2)

    wall_time_sec = float(time.perf_counter() - run_start)
    stats = {
        "dataset": data_path.stem,
        "data_path": str(data_path),
        "omics": args.omics or "all",
        "base_model": args.base_model,
        "C": args.C,
        "class_weight": args.class_weight,
        "n_neurons_requested": args.n_neurons,
        "n_neurons_built": max_built,
        "cv_folds": args.cv_folds,
        "cv_repeats": args.cv_repeats,
        "cv_total_folds": total_folds,
        "jobs": args.jobs,
        "wall_time_sec": round(wall_time_sec, 6),
        "max_rss_gb": round(float(max_rss_bytes_from_rusage() / (1024 ** 3)), 6),
        "run_dir": str(out_dir),
        "raw_predictions_path": str(out_dir / "raw_predictions.csv"),
        "neuron_summary_path": str(out_dir / "neuron_summary.csv"),
        "selected_features_path": str(out_dir / "selected_features.csv"),
        "class_labels_path": str(out_dir / "class_labels.json"),
        "ground_truth_metadata_path": None if ground_truth is None else ground_truth["_path"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    stats_path = save_run_stats_json(out_dir, stats)
    summary_path = update_run_stats_summary(data_path.stem)
    print(f"saved raw bundle dir: {out_dir}")
    print(f"saved run stats: {stats_path}")
    print(f"updated stats summary: {summary_path}")
    return out_dir
