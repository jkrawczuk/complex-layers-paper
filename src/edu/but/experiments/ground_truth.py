import json
from pathlib import Path


def find_ground_truth_metadata(data_path: Path):
    candidates = [
        data_path.with_suffix(".metadata.json"),
        data_path.with_name(f"{data_path.stem}_metadata.json"),
        data_path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_ground_truth_metadata(data_path: Path):
    metadata_path = find_ground_truth_metadata(data_path)
    if metadata_path is None:
        return None

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    informative = metadata.get("informative_features", [])
    informative_weights = metadata.get("informative_weights", [])
    if informative and informative_weights and len(informative) != len(informative_weights):
        raise ValueError(
            f"Ground-truth metadata mismatch in {metadata_path}: "
            "informative_features and informative_weights must have the same length."
        )

    metadata["_path"] = str(metadata_path)
    metadata["_informative_feature_set"] = set(int(v) for v in informative)
    metadata["_informative_weight_map"] = {
        int(idx): float(weight) for idx, weight in zip(informative, informative_weights)
    }
    metadata["_total_informative_weight"] = float(sum(abs(v) for v in informative_weights))
    return metadata


def summarize_feature_recovery(selected_features, ground_truth):
    selected_set = set(int(v) for v in selected_features)
    informative_set = ground_truth["_informative_feature_set"]
    weight_map = ground_truth["_informative_weight_map"]
    total_weight = ground_truth["_total_informative_weight"]

    recovered_informative = selected_set & informative_set
    recovered_weight = float(sum(abs(weight_map[idx]) for idx in recovered_informative))
    selected_count = len(selected_set)
    informative_count = len(recovered_informative)
    noise_count = max(0, selected_count - informative_count)
    total_informative = max(1, len(informative_set))

    precision = informative_count / selected_count if selected_count else 0.0
    recall = informative_count / total_informative
    noise_fraction = noise_count / selected_count if selected_count else 0.0
    weighted_recall = recovered_weight / total_weight if total_weight > 0 else 0.0

    return {
        "selected_count": float(selected_count),
        "informative_selected": float(informative_count),
        "noise_selected": float(noise_count),
        "precision": float(precision),
        "recall": float(recall),
        "noise_fraction": float(noise_fraction),
        "weighted_recall": float(weighted_recall),
    }
