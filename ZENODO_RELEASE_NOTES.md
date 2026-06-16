# Zenodo Release Notes

Release tag: `v1.0.0-peerj-submission`

## Main Archive

The GitHub release archive contains the tracked repository files:

- source code under `src/`;
- reproducibility scripts under `scripts/`;
- input datasets under `data/`;
- manuscript source under `paper/`;
- final manuscript figures under `paper/figures/`;
- release metadata files including `CITATION.cff`, `.zenodo.json`, `DATA_AVAILABILITY.md`, and
  `DATA_LICENSES.md`.

## Separate Zenodo Artifact

The generated paper-run result bundles are packaged as a separate Zenodo file:

- `complex-layers-paper-runs-v1.0.0.zip`

This artifact contains `results/paper_runs/`, including raw predictions, selected features, per-split
metrics, aggregate metrics, run metadata, and generated diagnostic figures for the manuscript experiments.

Current local artifact details:

- path: `zenodo-artifacts/complex-layers-paper-runs-v1.0.0.zip`;
- size: approximately 14 MB;
- ZIP entries: 257;
- SHA-256: `56db220cd5f7bd1b9979277c02bc38082aac2007de3b6f1538ff03f8201e05e7`.

## License Notes

The repository source code is MIT licensed. The synthetic dataset is distributed with the repository unless a
later release states otherwise. The processed `data/colon_cancer.csv` file corresponds to the Bioconductor
`colonCA` experiment data package (DOI: `10.18129/B9.bioc.colonCA`) and follows the upstream `colonCA` LGPL
package license, not the repository MIT code license.

## DOI Updates

After Zenodo DOI assignment, update DOI references in:

- `CITATION.cff`;
- `README.md`;
- `DATA_AVAILABILITY.md`;
- `paper/main.tex`.
