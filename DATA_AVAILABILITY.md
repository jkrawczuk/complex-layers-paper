# Data Availability Notes

This file records the data artifacts currently present in the repository and the checks needed before public
release.

## Included Data

### `data/synthetic_signal.csv`

Synthetic binary classification dataset. The accompanying `data/synthetic_signal.metadata.json` records the generation parameters, informative feature
indices, redundant feature indices, noise feature indices, and feature-strength ranking.

### `data/colon_cancer.csv`

Colon cancer microarray benchmark used in the manuscript experiments.

Before public release, confirm and document:

- original source and citation,
- original license or redistribution terms,
- whether the exact CSV in this repository can be redistributed,
- preprocessing steps, if any, used to produce this file.

If redistribution is not permitted, remove `data/colon_cancer.csv` from the public archive and replace it with
instructions for obtaining and preparing the same dataset version.

## PeerJ Submission Text Template

Use this only after replacing placeholders:

> The source code, experiment scripts, synthetic dataset, generated result bundles, and manuscript source are
> available in the archived repository at [Zenodo DOI]. The synthetic dataset generation parameters and
> ground-truth informative features are recorded in `data/synthetic_signal.metadata.json`. The colon cancer microarray benchmark was
> obtained from [source/citation] and prepared as `data/colon_cancer.csv` using [preprocessing description].
