# PeerJ manuscript skeleton

Ten katalog zawiera zalazek artykulu zgodny z template PeerJ (`wlpeerj`).

## Pliki
- `main.tex` - glowny plik manuskryptu
- `sections/*.tex` - sekcje artykulu
- `references.bib` - bibliografia

## Wymagany plik klasy
Do kompilacji potrzebny jest `wlpeerj.cls`.

Pobranie:
1. Otworz template PeerJ na Overleaf.
2. Kliknij `Open as Template`.
3. Pobierz source i skopiuj `wlpeerj.cls` do `paper/`.

## Kompilacja lokalna (przyklad)
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
