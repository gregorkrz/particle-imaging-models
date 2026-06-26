# Minimal makefile for Sphinx documentation
#
# Usage:
#   make html      # build the static site into build/html
#   make clean     # remove the build directory
#   make serve     # build, then serve at http://localhost:8000
#   make linkcheck # check external links

SPHINXOPTS    ?= --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PORT          ?= 8000
PYTHON        ?= python

.PHONY: help gen html clean serve linkcheck Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Regenerate the per-registry API pages from the live registries. Needs an env
# that can `import pimm`.
gen:
	@$(PYTHON) gen_api.py

html: gen
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "\nBuild finished. Open $(BUILDDIR)/html/index.html"

clean:
	rm -rf "$(BUILDDIR)" "$(SOURCEDIR)/api/registry" \
	       "$(SOURCEDIR)/api/generated" "$(SOURCEDIR)/api/*/generated"

serve: html
	@echo "Serving on http://localhost:$(PORT)  (override with: make serve PORT=8234)"
	@cd "$(BUILDDIR)/html" && python -m http.server $(PORT) --bind 127.0.0.1

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(O)

# Catch-all target: route unknown targets to sphinx-build.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
