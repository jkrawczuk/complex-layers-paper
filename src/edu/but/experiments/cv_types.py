from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    omics: str
    base_model: str
    C: float
    n_neurons: int
    random_state: int
    verbose: int
    cv_folds: int
    cv_repeats: int
    jobs: int


@dataclass(frozen=True)
class SplitTask:
    split_idx: int
    repeat_idx: int
    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    run_cfg: RunConfig

