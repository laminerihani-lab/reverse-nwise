# Build the paper.
#   make          -> main.pdf         (named build, for arXiv)
#   make anon     -> main-blind.pdf   (anonymised build, for ICST/AST double-blind)
#   make tables   -> regenerate LaTeX tables from result JSONs
#   make clean    -> remove build artifacts

MAIN = main
RESULTS = ../results

.PHONY: all anon tables clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib tables/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Anonymised (double-blind) build. Uses a separate jobname so it does not clobber
# the named main.pdf; produces main-blind.pdf.
anon:
	latexmk -pdf -interaction=nonstopmode -halt-on-error \
		-jobname=$(MAIN)-blind \
		-usepretex='\def\ANONYMOUS{}' $(MAIN).tex

# Regenerate all LaTeX tables from the raw experiment artifacts.
tables:
	cd .. && python -m analysis.tables \
		--exp04 $(RESULTS)/exp04_30runs.json \
		--exp03 $(RESULTS)/exp03_seed0.json \
		--exp02 $(RESULTS)/exp02_multidomain.json \
		--exp05 $(RESULTS)/exp05_seed0.json \
		--exp06 $(RESULTS)/exp06_seed0.json \
		--out paper/tables

clean:
	latexmk -C $(MAIN).tex
	latexmk -C -jobname=$(MAIN)-blind $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN)-blind.bbl $(MAIN).run.xml $(MAIN)-blind.run.xml

