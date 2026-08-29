# Build the paper.  `make` -> main.pdf ; `make tables` regenerates tables from
# result JSONs ; `make clean` removes build artifacts.

MAIN = main
RESULTS = ../results

.PHONY: all tables clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib tables/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Regenerate all LaTeX tables from the raw experiment artifacts.
tables:
	cd .. && python -m analysis.tables \
		--exp04 $(RESULTS)/exp04_30runs.json \
		--exp03 $(RESULTS)/exp03_seed0.json \
		--exp02 $(RESULTS)/exp02_multidomain.json \
		--exp06 $(RESULTS)/exp06_seed0.json \
		--out paper/tables

clean:
	latexmk -C $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml
