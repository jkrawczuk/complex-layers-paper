# Data Licenses And Provenance

This repository contains both generated data and a redistributed public benchmark dataset. Dataset licensing is
tracked separately from the MIT license used for the source code.

## `data/synthetic_signal.csv`

Synthetic dataset generated for this study. The generation parameters and ground-truth feature metadata are
stored in `data/synthetic_signal.metadata.json`.

License: MIT, same as the repository code, unless a later archived release states otherwise.

## `data/colon_cancer.csv`

Processed copy of the Alon et al. colon cancer microarray benchmark:

> Alon, U. et al. (1999). Broad patterns of gene expression revealed by clustering analysis of tumor and
> normal colon tissues probed by oligonucleotide arrays. Proceedings of the National Academy of Sciences,
> 96(12), 6745-6750.

The dataset is available through the Bioconductor experiment data package `colonCA`, DOI
`10.18129/B9.bioc.colonCA`, which lists the package license as LGPL. The local CSV contains 62 samples, 2000
gene-expression features, and one class-label column. Feature values are the version used by the manuscript
experiments, stored in a plain CSV format for Python reproducibility.

License/provenance note: treat this dataset as redistributed under the terms of the Bioconductor `colonCA`
package license, not under the repository MIT code license. Cite both Alon et al. (1999) and the archived
repository when using the processed CSV.

For Zenodo and other archival releases, the license separation is:

- source code: MIT License;
- `data/synthetic_signal.csv` and its metadata: MIT License unless the release states otherwise;
- `data/colon_cancer.csv`: processed redistribution of the Bioconductor `colonCA` data under the upstream
  LGPL package license.
