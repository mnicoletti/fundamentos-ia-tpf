# Makefile — TP Fundamentos de la IA
#
# Targets:
#   make build     -> build_nb.py genera dist/TP_Fundamentos_IA_Financiero.ipynb (SIN outputs)
#   make verify    -> depende de build; corre verify_pipeline.py (smoke test)
#   make pdf       -> nbconvert --to webpdf --no-input sobre lo que haya en dist/ ahora mismo
#                     (requiere Chromium via Playwright)
#   make pdf-html  -> igual que pdf, fallback si webpdf falla (ADR-004): HTML + print-to-PDF manual
#   make clean     -> borra dist/ y el entorno virtual local
#
# IMPORTANTE (F4/F5): pdf y pdf-html NO dependen de build a propósito. Una vez
# congelada la corrida canónica (dist/TP_Fundamentos_IA_Financiero.ipynb con
# outputs reales, ADR-001), correr "make build" la pisa con una versión vacía
# sin ejecutar. En F5 (typographic pass del PDF) siempre se corre sobre el
# .ipynb congelado que ya está en dist/, nunca se reconstruye antes.
#
# Requiere python3.11 disponible en PATH. El resto de las dependencias se
# instalan en un virtualenv local (.venv), gitignored.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

VENV       := .venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
STAMP      := $(VENV)/.deps-installed
PDF_STAMP  := $(VENV)/.pdf-deps-installed

NOTEBOOK      := dist/TP_Fundamentos_IA_Financiero.ipynb
PDF_BASENAME  := TP_Fundamentos_IA_Exposicion

.PHONY: build verify pdf pdf-html clean

$(STAMP): requirements.txt
	python3.11 -m venv $(VENV)
	$(PIP) install --upgrade pip -q
	$(PIP) install -q -r requirements.txt
	touch $(STAMP)

build: $(STAMP)
	$(PYTHON) build_nb.py

verify: build
	$(PYTHON) verify_pipeline.py

$(PDF_STAMP): $(STAMP) requirements-pdf.txt
	$(PIP) install -q -r requirements-pdf.txt
	$(PYTHON) -m playwright install chromium
	touch $(PDF_STAMP)

pdf: $(PDF_STAMP)
	$(PYTHON) -m nbconvert --to webpdf --no-input \
		--TagRemovePreprocessor.enabled=True \
		--TagRemovePreprocessor.remove_cell_tags='["no-pdf"]' \
		--output-dir dist --output $(PDF_BASENAME) \
		$(NOTEBOOK)

pdf-html: $(STAMP)
	$(PYTHON) -m nbconvert --to html --no-input \
		--TagRemovePreprocessor.enabled=True \
		--TagRemovePreprocessor.remove_cell_tags='["no-pdf"]' \
		--output-dir dist --output $(PDF_BASENAME) \
		$(NOTEBOOK)
	@echo "HTML generado en dist/$(PDF_BASENAME).html — abrir en el navegador e imprimir a PDF (fallback ADR-004)."

clean:
	rm -rf dist $(VENV)
