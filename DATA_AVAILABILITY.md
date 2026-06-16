# Data Availability Notes

This file records the data artifacts currently present in the repository and the checks needed before public
release.

## Included Data

### `data/synthetic_signal.csv`

Synthetic binary classification dataset. The accompanying `data/synthetic_signal.metadata.json` records the generation parameters, informative feature
indices, redundant feature indices, noise feature indices, and feature-strength ranking.

### `data/colon_cancer.csv`

Processed colon cancer microarray benchmark used in the manuscript experiments. The dataset corresponds to the
Alon et al. (1999) colon cancer microarray benchmark and is available through the Bioconductor experiment data
package `colonCA` (DOI: `10.18129/B9.bioc.colonCA`), which lists the package license as LGPL.

The local CSV is retained for exact reproducibility of the Python experiments. Treat it as a redistributed data
artifact under the upstream `colonCA` package license, not under the repository MIT code license.

For archival metadata, state the license separation explicitly: source code is MIT licensed, while
`data/colon_cancer.csv` is redistributed as a processed form of the Bioconductor `colonCA` dataset under the
upstream LGPL package license.

## PeerJ Submission Text Template

Use this only after replacing placeholders:

> The source code, experiment scripts, synthetic dataset, final manuscript figure assets, and manuscript source
> are available in the archived repository at [Zenodo DOI]. The synthetic dataset generation parameters and
> ground-truth informative features are recorded in `data/synthetic_signal.metadata.json`. The colon cancer
> microarray benchmark corresponds to the Alon et al. (1999) dataset distributed by the Bioconductor `colonCA`
> experiment data package (DOI: `10.18129/B9.bioc.colonCA`; LGPL) and is included as
> `data/colon_cancer.csv` in the processed CSV form used for the experiments. The repository source code is
> MIT licensed; `data/colon_cancer.csv` is redistributed under the upstream `colonCA` LGPL package license,
> not under the repository MIT license.
