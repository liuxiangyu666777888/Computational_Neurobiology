# NeurIPS 2026-Style Report

This folder contains a standalone English report package using the official NeurIPS 2026 LaTeX style file.

Files:

```text
main.tex            main LaTeX report
neurips_2026.sty    official NeurIPS 2026 style file
references.bib      BibTeX references
figures/            copied figures used by the report
```

No checklist is included because this is a course report, not a NeurIPS submission.

Compile with XeLaTeX:

```bash
latexmk -xelatex main.tex
```

Or manually:

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

The report uses the completed full-data experiment results. It does not require rerunning training.
