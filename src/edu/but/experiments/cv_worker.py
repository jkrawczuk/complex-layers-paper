import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from edu.but.experiments.ground_truth import summarize_feature_recovery
from edu.but.layers.complex import ComplexL1NeuronsClassifier

_PAR_X = None
_PAR_Y = None


def features_for_neuron(counts, n_neuron):
    if n_neuron < 1:
        return None
    if len(counts) < n_neuron:
        return None
    return float(counts[n_neuron - 1])


def objective_for_neuron(objectives, n_neuron):
    if n_neuron < 1:
        return None
    if len(objectives) < n_neuron:
        return None
    value = objectives[n_neuron - 1]
    if value is None:
        return None
    return value


def class_counts_as_str(y_values):
    labels, counts = np.unique(y_values, return_counts=True)
    return ";".join(f"{label}:{int(count)}" for label, count in zip(labels, counts))


def minority_class_label(y_values):
    labels, counts = np.unique(y_values, return_counts=True)
    if len(labels) == 0:
        raise ValueError("Expected at least one class label.")
    return labels[np.argmin(counts)]


def classification_metrics(y_true, y_pred, minority_label):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "minority_precision": float(
            precision_score(y_true, y_pred, pos_label=minority_label, zero_division=0)
        ),
        "minority_recall": float(
            recall_score(y_true, y_pred, pos_label=minority_label, zero_division=0)
        ),
        "minority_f1": float(f1_score(y_true, y_pred, pos_label=minority_label, zero_division=0)),
    }


def init_worker(X, y):
    global _PAR_X, _PAR_Y
    _PAR_X = X
    _PAR_Y = y


def run_single_split(task):
    split_idx = task["split_idx"]
    repeat_idx = task["repeat_idx"]
    fold_idx = task["fold_idx"]
    train_idx = task["train_idx"]
    test_idx = task["test_idx"]
    cfg = task["cfg"]

    X = _PAR_X
    y = _PAR_Y
    if X is None or y is None:
        raise RuntimeError("Worker data not initialized.")

    fit_start = time.perf_counter()
    clf = ComplexL1NeuronsClassifier(
        n_neurons=cfg["n_neurons"],
        base_model=cfg["base_model"],
        C=cfg["C"],
        class_weight_mode=cfg.get("class_weight_mode", "balanced"),
        max_iter=5000,
        random_state=cfg["random_state"],
        verbose=cfg["verbose"],
    )
    clf.fit(X[train_idx], y[train_idx])
    fit_time_sec = float(time.perf_counter() - fit_start)

    P_test = clf.vote_matrix(X[test_idx])
    P_train = clf.vote_matrix(X[train_idx])
    built_n = int(P_test.shape[1])
    counts_full = [len(g) for g in clf.get_feature_groups()]
    feature_groups_full = [np.asarray(g, dtype=int) for g in clf.get_feature_groups()]
    input_feature_groups_full = [
        np.asarray(g, dtype=int) for g in clf.get_input_feature_groups()
    ]
    objectives_full = [
        getattr(n, "common_objective_", getattr(n, "final_objective_", None))
        for n in clf.neurons_
    ]
    gt = cfg.get("ground_truth")

    y_train = y[train_idx]
    y_test = y[test_idx]
    y_train_enc = clf._label_encoder.transform(y_train)
    y_test_enc = clf._label_encoder.transform(y_test)
    minority_label = minority_class_label(y_train)
    class_counts_train = class_counts_as_str(y_train)
    class_counts_test = class_counts_as_str(y_test)

    per_n_rows = []
    raw_prediction_rows = []
    neuron_summary_rows = []
    selected_feature_rows = []
    raw_by_set_sample = {}
    for sample_index, y_true in zip(train_idx, y_train_enc):
        raw_by_set_sample[("train", int(sample_index))] = {
            "split_id": split_idx,
            "repeat_id": repeat_idx,
            "fold_id": fold_idx,
            "set": "train",
            "sample_index": int(sample_index),
            "y_true": int(y_true),
        }
    for sample_index, y_true in zip(test_idx, y_test_enc):
        raw_by_set_sample[("test", int(sample_index))] = {
            "split_id": split_idx,
            "repeat_id": repeat_idx,
            "fold_id": fold_idx,
            "set": "test",
            "sample_index": int(sample_index),
            "y_true": int(y_true),
        }
    pred_n1 = None
    pred_n2 = None
    cumulative_selected = set()
    for n in range(1, min(cfg["n_neurons"], built_n) + 1):
        y_pred_test = clf.predict_with_votes(P_test, n)
        y_pred_train = clf.predict_with_votes(P_train, n)
        vote_test_metrics = classification_metrics(y_test, y_pred_test, minority_label)
        vote_train_metrics = classification_metrics(y_train, y_pred_train, minority_label)

        neuron_pred_test = P_test[:, n - 1].astype(int)
        neuron_pred_train = P_train[:, n - 1].astype(int)
        neuron_pred_labels_test = clf._label_encoder.inverse_transform(neuron_pred_test)
        neuron_pred_labels_train = clf._label_encoder.inverse_transform(neuron_pred_train)
        neuron_test_metrics = classification_metrics(y_test, neuron_pred_labels_test, minority_label)
        neuron_train_metrics = classification_metrics(y_train, neuron_pred_labels_train, minority_label)

        objective_n = objective_for_neuron(objectives_full, n)
        loss_n = objective_n[0] if isinstance(objective_n, tuple) and len(objective_n) > 0 else None
        l1_n = objective_n[1] if isinstance(objective_n, tuple) and len(objective_n) > 1 else None
        feature_n = features_for_neuron(counts_full, n)
        selected_n = feature_groups_full[n - 1]
        cumulative_selected.update(int(v) for v in selected_n.tolist())

        recovery_n = None
        cumulative_recovery = None
        if gt is not None:
            recovery_n = summarize_feature_recovery(selected_n, gt)
            cumulative_recovery = summarize_feature_recovery(sorted(cumulative_selected), gt)

        for sample_index, y_true, y_pred in zip(
            train_idx,
            y_train_enc,
            neuron_pred_train,
        ):
            raw_by_set_sample[("train", int(sample_index))][f"pred_n{n}"] = int(y_pred)
        for sample_index, y_true, y_pred in zip(
            test_idx,
            y_test_enc,
            neuron_pred_test,
        ):
            raw_by_set_sample[("test", int(sample_index))][f"pred_n{n}"] = int(y_pred)

        neuron_summary_rows.append(
            {
                "split_id": split_idx,
                "repeat_id": repeat_idx,
                "fold_id": fold_idx,
                "n_neuron": n,
                "features_nth_neuron": feature_n,
                "informative_selected_nth": None
                if recovery_n is None
                else recovery_n["informative_selected"],
                "noise_selected_nth": None if recovery_n is None else recovery_n["noise_selected"],
                "feature_precision_nth": None if recovery_n is None else recovery_n["precision"],
                "feature_recall_nth": None if recovery_n is None else recovery_n["recall"],
                "feature_weight_recall_nth": None
                if recovery_n is None
                else recovery_n["weighted_recall"],
                "noise_fraction_nth": None if recovery_n is None else recovery_n["noise_fraction"],
                "cumulative_informative_selected": None
                if cumulative_recovery is None
                else cumulative_recovery["informative_selected"],
                "cumulative_feature_precision": None
                if cumulative_recovery is None
                else cumulative_recovery["precision"],
                "cumulative_feature_recall": None
                if cumulative_recovery is None
                else cumulative_recovery["recall"],
                "cumulative_feature_weight_recall": None
                if cumulative_recovery is None
                else cumulative_recovery["weighted_recall"],
                "cumulative_noise_fraction": None
                if cumulative_recovery is None
                else cumulative_recovery["noise_fraction"],
                "loss": loss_n,
                "l1_norm": l1_n,
            }
        )

        neuron = clf.neurons_[n - 1]
        coef = np.asarray(getattr(neuron, "coef_", []), dtype=float).reshape(-1)
        input_features = input_feature_groups_full[n - 1]
        if coef.size:
            coef_by_feature = {
                int(feature): float(coef[local_idx])
                for local_idx, feature in enumerate(input_features)
                if local_idx < coef.size
            }
        else:
            coef_by_feature = {}
        for feature_index in selected_n:
            selected_feature_rows.append(
                {
                    "split_id": split_idx,
                    "repeat_id": repeat_idx,
                    "fold_id": fold_idx,
                    "n_neuron": n,
                    "feature_index": int(feature_index),
                    "coefficient": coef_by_feature.get(int(feature_index), ""),
                }
            )

        per_n_rows.append(
            {
                "n_neurons": n,
                "minority_class": str(minority_label),
                "vote_accuracy": vote_test_metrics["accuracy"],
                "vote_balanced_accuracy": vote_test_metrics["balanced_accuracy"],
                "vote_minority_precision": vote_test_metrics["minority_precision"],
                "vote_minority_recall": vote_test_metrics["minority_recall"],
                "vote_minority_f1": vote_test_metrics["minority_f1"],
                "vote_train_accuracy": vote_train_metrics["accuracy"],
                "vote_train_balanced_accuracy": vote_train_metrics["balanced_accuracy"],
                "vote_train_minority_precision": vote_train_metrics["minority_precision"],
                "vote_train_minority_recall": vote_train_metrics["minority_recall"],
                "vote_train_minority_f1": vote_train_metrics["minority_f1"],
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
                "features_nth_neuron": feature_n,
                "informative_selected_nth": None if recovery_n is None else recovery_n["informative_selected"],
                "noise_selected_nth": None if recovery_n is None else recovery_n["noise_selected"],
                "feature_precision_nth": None if recovery_n is None else recovery_n["precision"],
                "feature_recall_nth": None if recovery_n is None else recovery_n["recall"],
                "feature_weight_recall_nth": None
                if recovery_n is None
                else recovery_n["weighted_recall"],
                "noise_fraction_nth": None if recovery_n is None else recovery_n["noise_fraction"],
                "cumulative_informative_selected": None
                if cumulative_recovery is None
                else cumulative_recovery["informative_selected"],
                "cumulative_feature_precision": None
                if cumulative_recovery is None
                else cumulative_recovery["precision"],
                "cumulative_feature_recall": None
                if cumulative_recovery is None
                else cumulative_recovery["recall"],
                "cumulative_feature_weight_recall": None
                if cumulative_recovery is None
                else cumulative_recovery["weighted_recall"],
                "cumulative_noise_fraction": None
                if cumulative_recovery is None
                else cumulative_recovery["noise_fraction"],
                "loss": loss_n,
                "l1_norm": l1_n,
                "split_id": split_idx,
                "repeat_id": repeat_idx,
                "fold_id": fold_idx,
                "n_neurons_requested": cfg["n_neurons"],
                "n_neurons_built_in_split": built_n,
                "y_train_size": int(len(train_idx)),
                "y_test_size": int(len(test_idx)),
                "class_counts_train": class_counts_train,
                "class_counts_test": class_counts_test,
            }
        )
        if n == 1:
            pred_n1 = y_pred_test.tolist()
        elif n == 2:
            pred_n2 = y_pred_test.tolist()

    split_summary = {
        "split_id": split_idx,
        "repeat_id": repeat_idx,
        "fold_id": fold_idx,
        "n_neurons_requested": cfg["n_neurons"],
        "n_neurons_built_in_split": built_n,
        "fit_time_sec": round(fit_time_sec, 6),
        "minority_class": str(minority_label),
        "y_train_size": int(len(train_idx)),
        "y_test_size": int(len(test_idx)),
        "class_counts_train": class_counts_train,
        "class_counts_test": class_counts_test,
        "class_label_0": str(clf._label_encoder.classes_[0]),
        "class_label_1": str(clf._label_encoder.classes_[1]),
    }
    for _, row in sorted(raw_by_set_sample.items(), key=lambda item: (item[0][0], item[0][1])):
        raw_prediction_rows.append(row)
    return {
        "split_id": split_idx,
        "repeat_id": repeat_idx,
        "fold_id": fold_idx,
        "built_n": built_n,
        "per_n_rows": per_n_rows,
        "split_summary": split_summary,
        "raw_prediction_rows": raw_prediction_rows,
        "neuron_summary_rows": neuron_summary_rows,
        "selected_feature_rows": selected_feature_rows,
        "pred_n1": pred_n1,
        "pred_n2": pred_n2,
    }
