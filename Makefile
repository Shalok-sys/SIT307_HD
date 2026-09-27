PY ?= python

all: results figures report

results:
	$(PY) part1_reproduce.py
	$(PY) part2_dpcs.py

figures: results
	$(PY) make_figs.py

report: figures
	pdflatex -interaction=nonstopmode report.tex
	pdflatex -interaction=nonstopmode report.tex

clean:
	rm -f *.aux *.log *.out results_part*.json fig*.png report.pdf

.PHONY: all results figures report clean
