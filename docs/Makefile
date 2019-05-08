# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# I got this code snippet from https://jamwheeler.com/college-productivity/how-to-write-beautiful-code-documentation/
.ONESHELL:
gh-pages:
	rm -rf /tmp/gh-pages
	cp -r $(BUILDDIR)/html /tmp/gh-pages
	git checkout gh-pages
	cd .. && rm -rf * && cp -r /tmp/gh-pages/* . && git add . && git commit -m "Updated gh-pages" && git push && git checkout master

view-docs:
	open ./build/html/index.html