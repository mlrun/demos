MLRUN_TAG ?=

.PHONY: help
help: ## Display available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all:
	$(error please pick a target)

.PHONY: release
release: ## Release a version
ifndef MLRUN_TAG
	$(error MLRUN_TAG is undefined)
endif
	TAG_SUFFIX=$$(echo $${MLRUN_TAG%.*}.x); \
	BRANCH_NAME=$$(echo release/$$TAG_SUFFIX); \
	git fetch origin $$BRANCH_NAME || EXIT_CODE=$$?; \
	echo $$EXIT_CODE; \
	if [ "$$EXIT_CODE" = "0" ]; \
		then \
			echo "Branch $$BRANCH_NAME exists. Adding changes"; \
			git checkout $$BRANCH_NAME; \
			git merge --squash $(MLRUN_TAG); \
		else \
			echo "Creating new branch: $$BRANCH_NAME"; \
			git checkout --orphan $$BRANCH_NAME; \
	fi; \
	git commit -m "Adding $(MLRUN_TAG) tag contents"; \
	git push origin $$BRANCH_NAME
