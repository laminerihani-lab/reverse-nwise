# Build the AST 2027 paper (ACM acmart sigconf). Section bodies, references.bib,
# and tables are SHARED with the ICST build under ../paper (single source of
# truth); this folder only supplies the ACM class and front matter.
#
#   make          -> main.pdf         (named build)
#   make anon     -> main-blind.pdf   (anonymised build, AST double-blind)
#   make tables   -> regenerate the shared tables from result JSONs
#   make clean    -> remove build artifacts

MAIN = main
RESULTS = ../results

.PHONY: all anon tables clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex ../paper/sections/*.tex ../paper/references.bib ../paper/tables/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

anon:
	latexmk -pdf -interaction=nonstopmode -halt-on-error \
		-jobname=$(MAIN)-blind \
		-usepretex='\def\ANONYMOUS{}' $(MAIN).tex

# Regenerate the shared LaTeX tables (writes into ../paper/tables, used by both builds).
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
