# PYTHON is the interpreter used to regenerate this content. docs/user_guide/conf.py
# passes PYTHON=sys.executable so it is the same interpreter that is building the
# docs; the default here only applies when the Makefile is run by hand. Do not go
# back to a bare `python` -- resolved from PATH, it can regenerate the user guide
# from a different yapss installation than the one being documented, and succeed.
PYTHON ?= python

help: ## Print this help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[35m%-10s\033[0m %s\n", $$1, $$2}'

SRC_DIR := ../../../../src/yapss/examples

# What the output depends on besides the example itself: every module of the package but
# the other examples, which no example imports (the tox env installs a fresh copy of the
# tree on every run, so the source files are what the scripts import), the environment's
# other distributions, recorded by docs_env_stamp.py in a stamp rewritten only when they
# change, and the generator and this Makefile. A rule that misses an input leaves output
# from an older build in place, which is then what the local docs show, while Read the
# Docs, starting clean, shows something else.
YAPSS_FILES := $(shell find ../../../../src/yapss -name '*.py' -not -path '*/__pycache__/*' \
	-not -path '*/yapss/examples/*')
ENV_STAMP := .docs-env-stamp
$(shell $(PYTHON) ../../../../tools/docs_env_stamp.py $(ENV_STAMP))
DEPENDS := $(YAPSS_FILES) $(ENV_STAMP) make_plots.py Makefile

# One run of make_plots.py writes an example's text output and all of its figures, so the
# text file is the one target per example, and make runs each script exactly once, also
# under -j. Make starts the targets in this order, so the longest run comes first: the
# wall time of a parallel build is at least that of its longest job.
EXAMPLES := \
	delta_iii_ascent \
	minimum_time_to_climb \
	dynamic_soaring \
	goddard_problem_3_phase \
	brachistochrone \
	newton \
	orbit_raising \
	isoperimetric \
	rosenbrock \
	goddard_problem_1_phase \
	brachistochrone_minimal \
	hs071
TEXT := $(addsuffix .txt,$(EXAMPLES))

.PHONY: all
all: $(TEXT) ## Generate all figures and text files for the script examples.

# A failed run must not leave a partial text file newer than its inputs, which make would
# then take to be up to date.
.DELETE_ON_ERROR:

$(TEXT): %.txt: $(SRC_DIR)/%.py $(DEPENDS)
	$(PYTHON) make_plots.py $* > $@

.PHONY: clean
clean: ## Remove generated files and caches.
	@rm -rf .coverage htmlcov __pycache__ .mypy_cache .ipynb_checkpoints *.txt *.png $(ENV_STAMP)
