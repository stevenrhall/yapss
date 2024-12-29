.PHONY: help
# adapted from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Print this help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[35m%-10s\033[0m %s\n", $$1, $$2}'

# Directory paths
SRC_DIR = ../../../examples/notebooks
DEST_DIR = .
YAPSS_DIRS = ../../../src/yapss ../../../src/yapss/_private ../../../src/yapss/numpy
YAPSS_FILES = $(wildcard $(YAPSS_DIRS)/*.py)

# List of notebook files to generate
NOTEBOOKS = \
	tutorial.ipynb \
	rosenbrock.ipynb \
	hs071.ipynb \
	brachistochrone.ipynb \
	isoperimetric.ipynb \
	newton.ipynb \
	goddard_problem_1_phase.ipynb \
	goddard_problem_3_phase.ipynb \
	orbit_raising.ipynb \
	dynamic_soaring.ipynb \
	minimum_time_to_climb.ipynb \
	delta_iii_ascent.ipynb

.PHONY: all clean
all: $(NOTEBOOKS) ## Generate all Jupyter notebooks

# Generate each notebook with additional YAPSS_FILES as prerequisites
$(NOTEBOOKS): $(DEST_DIR)/%.ipynb: $(SRC_DIR)/%.ipynb $(YAPSS_FILES)
	jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --output-dir $(DEST_DIR) --output $(basename $@).ipynb $< > /dev/null

.PHONY: clean
clean: ## Remove Jupyter notebooks
	rm -f $(addprefix $(DEST_DIR)/, *.ipynb)
	rm -r -f $(addprefix $(DEST_DIR)/, .ipynb_checkpoints)
	rm -rf .jupyter