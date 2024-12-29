help: ## Print this help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[35m%-10s\033[0m %s\n", $$1, $$2}'

SRC_DIR := ../../../../src/yapss/examples
YAPSS_DIRS := ../../../../src/yapss ../../../../src/yapss/_private ../../../../src/yapss/numpy
YAPSS_FILES := $(wildcard $(YAPSS_DIRS)/*.py)

TEXT := \
	brachistochrone.txt \
	brachistochrone_minimal.txt \
	delta_iii_ascent.txt \
	dynamic_soaring.txt \
	goddard_problem_1_phase.txt \
	goddard_problem_3_phase.txt \
	hs071.txt \
	isoperimetric.txt \
	minimum_time_to_climb.txt \
	newton.txt \
	orbit_raising.txt \
	rosenbrock.txt

PNG := \
	brachistochrone_plot_1.png  brachistochrone_plot_2.png  brachistochrone_plot_3.png \
	brachistochrone_plot_4.png brachistochrone_plot_5.png \
	brachistochrone_minimal_plot_1.png \
	delta_iii_ascent_plot_1.png delta_iii_ascent_plot_2.png \
	delta_iii_ascent_plot_3.png delta_iii_ascent_plot_4.png \
	dynamic_soaring_plot_1.png dynamic_soaring_plot_2.png dynamic_soaring_plot_3.png \
	dynamic_soaring_plot_4.png dynamic_soaring_plot_5.png dynamic_soaring_plot_6.png \
	orbit_raising_plot_1.png orbit_raising_plot_2.png \
	orbit_raising_plot_3.png orbit_raising_plot_4.png \
	orbit_raising_plot_5.png \
	goddard_problem_1_phase_plot_1.png goddard_problem_1_phase_plot_2.png \
	goddard_problem_1_phase_plot_3.png goddard_problem_1_phase_plot_4.png \
	goddard_problem_3_phase_plot_1.png goddard_problem_3_phase_plot_2.png \
	goddard_problem_3_phase_plot_3.png goddard_problem_3_phase_plot_4.png \
	newton_plot_1.png  newton_plot_2.png newton_plot_3.png \
	minimum_time_to_climb_plot_1.png minimum_time_to_climb_plot_2.png \
	minimum_time_to_climb_plot_3.png minimum_time_to_climb_plot_4.png \
	minimum_time_to_climb_plot_5.png minimum_time_to_climb_plot_6.png \
	minimum_time_to_climb_plot_7.png



# Define all target to build every file individually
all: $(TEXT) $(PNG) ## Generate all figures and text files for the script examples.

# Define individual target rules with prerequisites
newton_plot_%.png: $(SRC_DIR)/newton.py $(YAPSS_FILES)
	python make_plots.py newton > newton.txt

brachistochrone_plot_%.png: $(SRC_DIR)/brachistochrone.py $(YAPSS_FILES)
	python make_plots.py brachistochrone > brachistochrone.txt

brachistochrone_minimal_plot_%.png: $(SRC_DIR)/brachistochrone_minimal.py $(YAPSS_FILES)
	python make_plots.py brachistochrone_minimal > brachistochrone_minimal.txt

dynamic_soaring_plot_%.png: $(SRC_DIR)/dynamic_soaring.py $(YAPSS_FILES)
	python make_plots.py dynamic_soaring > dynamic_soaring.txt

orbit_raising_plot_%.png: $(SRC_DIR)/orbit_raising.py $(YAPSS_FILES)
	python make_plots.py orbit_raising > orbit_raising.txt

goddard_problem_1_phase_plot_%.png: $(SRC_DIR)/goddard_problem_1_phase.py $(YAPSS_FILES)
	python make_plots.py goddard_problem_1_phase > goddard_problem_1_phase.txt

goddard_problem_3_phase_plot_%.png: $(SRC_DIR)/goddard_problem_3_phase.py $(YAPSS_FILES)
	python make_plots.py goddard_problem_3_phase > goddard_problem_3_phase.txt

isoperimetric_plot_%.png: $(SRC_DIR)/isoperimetric.py $(YAPSS_FILES)
	python make_plots.py isoperimetric > isoperimetric.txt

rosenbrock_plot_%.png: $(SRC_DIR)/rosenbrock.py $(YAPSS_FILES)
	python make_plots.py rosenbrock > rosenbrock.txt

minimum_time_to_climb_plot_%.png: $(SRC_DIR)/minimum_time_to_climb.py $(YAPSS_FILES)
	python make_plots.py minimum_time_to_climb > minimum_time_to_climb.txt

delta_iii_ascent_plot_%.png: $(SRC_DIR)/delta_iii_ascent.py $(YAPSS_FILES)
	python make_plots.py delta_iii_ascent > delta_iii_ascent.txt

# Each .txt file depends on $(YAPSS_FILES) so it rebuilds if any source changes
%.txt: $(SRC_DIR)/%.py $(YAPSS_FILES)
	python make_plots.py $(basename $@) > $@

.PHONY: clean
clean: ## Remove generated files and caches.
	@rm -rf .coverage htmlcov __pycache__ .mypy_cache .ipynb_checkpoints *.txt *.png
