.PHONY: help
# adapted from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Print this help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[35m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: execute
execute: $(wildcard *.ipynb) ## Execute all JupyterLab notebooks in place
	@for f in $(shell ls *.ipynb); do jupyter nbconvert --execute --inplace $${f}; done

.PHONY: clean
clean: $(wildcard *.ipynb) ## Clear output cells and remove kernel and language metadata from all JupyterLab notebooks
	nbwipers clean-all . -y
# --indent 1 is nbformat's own serialization, which is what Jupyter writes on save
# and what black writes when it reformats a cell. jq's default of 2 disagreed with
# both, so every notebook one of them touched came back reindented from end to end
# and a one-line change read as a diff of the whole file. At 1 the three agree
# byte for byte and nothing reindents anything.
	@for nb in *.ipynb; do \
		echo "Stripping kernel metadata from $$nb"; \
		jq --indent 1 'del(.metadata.kernelspec, .metadata.language_info)' "$$nb" > tmp_$$nb && mv tmp_$$nb "$$nb"; \
	done
	rm -rf .ipynb_checkpoints
