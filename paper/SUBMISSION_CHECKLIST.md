# PeerJ Computer Science Submission Checklist

This checklist records items that still need a human decision before final submission or Zenodo archiving.

## Manuscript

- [x] PeerJ LaTeX class included.
- [x] Line numbers enabled through the document class.
- [x] Abstract structured as Background, Methods, Results, and Conclusions.
- [x] Funding statement added.
- [x] AI assistance declaration added.
- [ ] Confirm final title.
- [ ] Confirm final author order, affiliations, and corresponding author metadata.
- [ ] Review manuscript language for one consistent English variant.
- [ ] Replace DOI placeholder in Data Availability after Zenodo archiving.

## Data And Code

- [x] Source code organized under `src/edu/but`.
- [x] Reproducibility scripts organized under `scripts/`.
- [x] Generated `results/` ignored by git.
- [x] Document redistribution basis for `data/colon_cancer.csv` through Bioconductor `colonCA`.
- [ ] Decide whether to attach generated result bundles as supplemental/Zenodo artifacts.
- [ ] Create the exact release intended for submission.
- [ ] Archive the release on Zenodo and update `CITATION.cff`, `README.md`, and manuscript Data Availability.

## Figures

- [x] Final figure assets included under `paper/figures/`.
- [x] Figure captions mention uniform vote, weighted vote, and single-neuron trajectories where applicable.
- [ ] Visually inspect final generated PDF before submission.
