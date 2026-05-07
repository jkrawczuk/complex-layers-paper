# Manuscript Build Notes

The manuscript uses the PeerJ `wlpeerj` LaTeX class included in this directory.

Recommended local build:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Clean auxiliary files after building:

```bash
latexmk -c main.tex
```

The generated `main.pdf` is intentionally ignored and should be produced from source when needed.
