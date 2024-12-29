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
	@for nb in *.ipynb; do \
		echo "Stripping kernel metadata from $$nb"; \
		jq 'del(.metadata.kernelspec, .metadata.language_info)' "$$nb" > tmp_$$nb && mv tmp_$$nb "$$nb"; \
	done
	rm -rf .ipynb_checkpoints
