.PHONY: help
# adapted from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Print this help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[35m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: execute
execute: $(wildcard *.ipynb) ## Execute all JupyterLab notebooks in place
	@for f in $(shell ls *.ipynb); do jupyter nbconvert --execute --inplace $${f}; done

.PHONY: clean
clean: $(wildcard *.ipynb) ## Clear output cells and remove kernel and language metadata from all JupyterLab notebooks
# What is stripped is set in pyproject.toml's [tool.nbwipers], which the pre-commit hook
# reads too, so a notebook this target cleaned is one the hook accepts.
	nbwipers clean-all . -y
	rm -rf .ipynb_checkpoints
