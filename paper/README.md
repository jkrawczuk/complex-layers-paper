# Paper Workspace

This directory contains the PeerJ Computer Science manuscript source.

Main files:

- `main.tex` - top-level manuscript file.
- `sections/*.tex` - manuscript sections.
- `references.bib` - bibliography.
- `figures/` - final figure assets used by the manuscript.
- `wlpeerj.cls` - PeerJ LaTeX class file included for local compilation.

Build locally with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Generated PDF and LaTeX auxiliary files are ignored by git.
