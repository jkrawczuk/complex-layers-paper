#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def build_dataset(
    n_samples: int = 100,
    n_features: int = 2000,
    n_blocks: int = 3,
    block_size: int = 4,
    label_noise: float = 0.55,
    feature_noise: float = 0.2,
    noise_feature_scale: float = 1.0,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    n_informative = n_blocks * block_size
    n_noise = n_features - n_informative
    if n_noise < 0:
        raise ValueError("n_features must be at least n_blocks * block_size.")

    block_strengths = np.asarray([1.2, 0.9, 0.6], dtype=float)
    if n_blocks != len(block_strengths):
        raise ValueError("This generator currently defines exactly three latent signal blocks.")
    within_block_weights = np.asarray([1.4, -1.15, 0.9, -0.65], dtype=float)
    if block_size != len(within_block_weights):
        raise ValueError("This generator currently defines exactly four informative features per block.")

    latent = rng.normal(size=(n_samples, n_blocks))
    score = latent @ block_strengths + rng.normal(scale=label_noise, size=n_samples)
    labels = np.where(score > np.median(score), "positive", "negative")

    X = np.empty((n_samples, n_features), dtype=float)
    informative_weights = []
    informative_groups = []
    col = 0
    for block_idx, strength in enumerate(block_strengths):
        group = []
        for weight in within_block_weights * strength:
            X[:, col] = weight * latent[:, block_idx] + rng.normal(
                scale=feature_noise,
                size=n_samples,
            )
            informative_weights.append(float(weight))
            group.append(col)
            col += 1
        informative_groups.append(group)

    X[:, col:] = rng.normal(scale=noise_feature_scale, size=(n_samples, n_noise))

    metadata = {
        "dataset_name": "synthetic_signal",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "n_noise": n_noise,
        "n_blocks": n_blocks,
        "block_size": block_size,
        "label_noise": label_noise,
        "feature_noise": feature_noise,
        "noise_feature_scale": noise_feature_scale,
        "random_state": random_state,
        "label_names": ["negative", "positive"],
        "informative_features": list(range(n_informative)),
        "noise_features": list(range(n_informative, n_features)),
        "informative_weights": informative_weights,
        "block_strengths": block_strengths.tolist(),
        "informative_feature_groups": informative_groups,
        "ranking_by_absolute_weight": [
            int(idx)
            for idx in np.argsort(-np.abs(np.asarray(informative_weights)))
        ],
        "standardized_in_csv": False,
        "description": (
            "Synthetic high-dimensional binary classification dataset with three latent signal blocks, "
            "known informative features, and independent noise features."
        ),
    }
    return X, labels, metadata


def write_csv(path: Path, X, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row, label in zip(X, labels):
            writer.writerow([f"{value:.8f}" for value in row] + [label])


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic signal benchmark.")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_signal.csv"))
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("data/synthetic_signal.metadata.json"),
    )
    args = parser.parse_args()

    X, labels, metadata = build_dataset()
    write_csv(args.out, X, labels)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_out.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    unique, counts = np.unique(labels, return_counts=True)
    print(
        "Generated synthetic dataset:",
        f"path={args.out}",
        f"samples={X.shape[0]}",
        f"features={X.shape[1]}",
        f"class_counts={dict(zip(unique.tolist(), counts.astype(int).tolist()))}",
    )


if __name__ == "__main__":
    main()
