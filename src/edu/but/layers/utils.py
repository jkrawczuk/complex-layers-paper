import csv
from pathlib import Path
from collections import Counter

import numpy as np


_KNOWN_OMICS = {"clinical", "cnv", "mirna", "mutation", "rna"}


def _normalize_omics(omics):
    if omics is None:
        return None
    if isinstance(omics, str):
        parts = [p.strip() for p in omics.split(",")]
    else:
        parts = []
        for item in omics:
            parts.extend(p.strip() for p in str(item).split(","))
    parts = [p for p in parts if p]
    if not parts:
        return None
    normalized = [p.lower() for p in parts]
    unknown = sorted(set(normalized) - _KNOWN_OMICS)
    if unknown:
        raise ValueError(f"Unknown omics: {unknown}. Known: {sorted(_KNOWN_OMICS)}")
    return normalized


def load_data(
    csv_path: Path,
    label_first: bool = False,
    has_header: bool = False,
    omics=None,
):
    X_rows = []
    y = []
    class_counts = Counter()
    omics = _normalize_omics(omics)
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = None
        if has_header:
            header = next(reader, None)
        if omics and not header:
            raise ValueError("--omics requires --has-header to select columns.")
        feature_indices = None
        n_features = None
        if header:
            total_cols = len(header)
            label_idx = 0 if label_first else total_cols - 1
            feature_indices = [i for i in range(total_cols) if i != label_idx]
            if omics:
                selected = []
                for i in feature_indices:
                    name = header[i].lower()
                    if any(name.endswith(f"_{o}") for o in omics):
                        selected.append(i)
                feature_indices = selected
            n_features = len(feature_indices)
        for row in reader:
            if not row:
                continue
            if feature_indices is None:
                if label_first:
                    label, *features = row
                else:
                    *features, label = row
                if n_features is None:
                    n_features = len(features)
            else:
                label = row[0] if label_first else row[-1]
                features = [row[i] for i in feature_indices]
            X_rows.append([float(v) for v in features])
            y.append(label)
            class_counts[label] += 1
    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y)
    omics_info = ",".join(omics) if omics else "all"
    n_samples = len(y)
    n_features = 0 if n_features is None else n_features
    print(
        "Loaded dataset:",
        f"path={csv_path}",
        f"samples={n_samples}",
        f"features={n_features}",
        f"omics={omics_info}",
        f"class_counts={dict(sorted(class_counts.items(), key=lambda x: str(x[0])))}",
    )
    return X, y
