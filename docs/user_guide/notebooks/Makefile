.PHONY: help
# adapted from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Print this help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[35m%-10s\033[0m %s\n", $$1, $$2}'

# Directory paths
SRC_DIR = ../../../examples/notebooks
DEST_DIR = .

# List of notebook files to generate. Make starts them in this order, so the longest run
# comes first: the wall time of a parallel build is at least that of its longest job.
NOTEBOOKS = \
	minimum_time_to_climb.ipynb \
	dynamic_soaring.ipynb \
	tutorial.ipynb \
	newton.ipynb \
	delta_iii_ascent.ipynb \
	rosenbrock.ipynb \
	goddard_problem_3_phase.ipynb \
	isoperimetric.ipynb \
	goddard_problem_1_phase.ipynb \
	orbit_raising.ipynb \
	brachistochrone.ipynb \
	hs071.ipynb

.PHONY: all clean
all: $(NOTEBOOKS) ## Generate all Jupyter notebooks

# PYTHON is the interpreter used to regenerate this content. docs/user_guide/conf.py
# passes PYTHON=sys.executable so it is the same interpreter that is building the
# docs; the default here only applies when the Makefile is run by hand. Do not go
# back to a bare `jupyter` -- resolved from PATH, it can regenerate the user guide
# from a different yapss installation than the one being documented, and succeed.
PYTHON ?= python

# What a notebook's output depends on besides the notebook itself: every module of the
# package but the examples (the tox env installs a fresh copy of the tree on every run, so
# the source files are what the notebooks import), the environment's other distributions,
# recorded by docs_env_stamp.py in a stamp rewritten only when they change, and this
# Makefile. A rule that misses an input leaves output from an older build in place, which
# is then what the local docs show, while Read the Docs, starting clean, shows something
# else.
YAPSS_FILES := $(shell find ../../../src/yapss -name '*.py' -not -path '*/__pycache__/*' \
	-not -path '*/yapss/examples/*')
ENV_STAMP := .docs-env-stamp
$(shell $(PYTHON) ../../../tools/docs_env_stamp.py $(ENV_STAMP))
DEPENDS := $(YAPSS_FILES) $(ENV_STAMP) Makefile

# The tutorial is the one notebook that imports an example script. A notebook that starts
# importing one needs the same line.
tutorial.ipynb: $(wildcard ../../../src/yapss/examples/*.py)

# ipykernel prints "Kernel is running over TCP without encryption" on every run;
# it's a false positive for a local, single-user build (loopback only). Setting
# --KernelManager.transport=ipc removes it but makes nbconvert hang (known
# jupyter/nbconvert issue), so we filter just that one line from stderr below.
# Everything else on stderr -- including real errors/tracebacks -- still prints,
# and the command's actual exit status is preserved so failures still break the build.
$(NOTEBOOKS): $(DEST_DIR)/%.ipynb: $(SRC_DIR)/%.ipynb $(DEPENDS)
	@tmp=$$(mktemp); \
	$(PYTHON) -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --output-dir $(DEST_DIR) --output $(basename $@).ipynb $< > /dev/null 2>"$$tmp"; \
	status=$$?; \
	grep -v "Kernel is running over TCP without encryption" "$$tmp" >&2; \
	rm -f "$$tmp"; \
	exit $$status

.PHONY: clean
clean: ## Remove Jupyter notebooks
	rm -f $(addprefix $(DEST_DIR)/, *.ipynb)
	rm -r -f $(addprefix $(DEST_DIR)/, .ipynb_checkpoints)
	rm -rf .jupyter $(ENV_STAMP)