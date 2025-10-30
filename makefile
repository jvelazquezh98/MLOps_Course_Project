#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = MLOps_Course_Project
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
FILE ?=

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 src
	isort --check --diff src
	black --check src

## Format source code with black
.PHONY: format
format:
	isort src
	black src

## Run tests
.PHONY: test
test:
	python -m pytest src/tests

## Add datasets to DVC and push to remote storage
.PHONY: dvc-add
dvc-add:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ ERROR: Debes especificar el archivo a versionar. Ejemplo:"; \
		echo "   make dvc-add FILE=data/processed/clientes_20251001.csv"; \
		exit 1; \
	fi
	@echo "🚀 Versionando archivo con DVC: $(FILE)"
	uv run dvc add $(FILE)
	git add $(FILE).dvc .gitignore
	git commit -m "Versionado automático de $(FILE)"
	@echo "⬆️ Subiendo dataset al remoto con DVC..."
	uv run dvc push
	@echo "✅ Archivo versionado y sincronizado con remoto: $(FILE)"

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)