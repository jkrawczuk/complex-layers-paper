# Manuscript Figures

This directory contains the final raster figure assets included by `paper/main.tex`.

Current manuscript panels:

- `colon_diagnostics_low.png`
- `colon_diagnostics_high.png`
- `synthetic_diagnostics_low.png`
- `synthetic_diagnostics_high.png`
- `synthetic_recovery_low.png`
- `synthetic_recovery_high.png`

The files are generated from local experiment bundles under `results/paper_runs_v2/` using:

```bash
python scripts/generate_paper_panel_figures.py
```

The `results/` directory is ignored by git. To regenerate these figures from scratch, first run the
experiment pipeline described in the repository-level `README.md`, then run the panel-generation script above.
