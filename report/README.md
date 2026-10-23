# AgentReel report (LaTeX, JAGI template)

Build with pdfLaTeX + BibTeX (Overleaf: upload this folder, compiler pdfLaTeX):

    pdflatex agentreel_report
    bibtex   agentreel_report
    pdflatex agentreel_report
    pdflatex agentreel_report

Files: `agentreel_report.tex`, `agentreel.bib`, `figures/c01-frames.png`, and the template files copied from
`../template/` (`jagi.sty`, `agi.bst`, `AGIS-logo-BW.png`). Needs TikZ and pgfplots (in TeX Live / MiKTeX / Overleaf).

To fill in before sharing: author names and e-mails (`\author`), and the journal-header fields in `\jagiheading`
(left as dashes: this is a lab report, not a journal submission).

All figures and numbers come from the C01 run export of 2026-10-22 (`export_C01-s42_20261022-1615`).
