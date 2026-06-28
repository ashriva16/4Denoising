# Makefile: simple .venv workflow for users (not developers)

VENV_DIR    := .venv
PYTHON      ?= python3.11
VENV_PYTHON := $(VENV_DIR)/bin/python
PIP         := $(VENV_DIR)/bin/pip

.PHONY: help env install dev test clean

## Show available commands for users
help:
	@echo "Commands for using this project:"
	@echo "  make env       - create $(VENV_DIR) and install dependencies"
	@echo "  make install   - reinstall/update dependencies into existing $(VENV_DIR)"
	@echo "  make dev       - install this project with test/development tools"
	@echo "  make test      - run the pytest smoke-test suite"
	@echo "  make clean     - remove cache/build artifacts (keep $(VENV_DIR))"

## Create a new virtual environment and install dependencies
env:
	@echo ">>> Creating virtual environment in $(VENV_DIR)"
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo ">>> Upgrading pip"
	@$(PIP) install --upgrade pip
	@if [ -f requirements.txt ]; then \
		echo ">>> Installing dependencies from requirements.txt"; \
		$(PIP) install -r requirements.txt; \
	else \
		echo "No requirements.txt found. Skipping dependency install."; \
	fi
	@echo
	@echo "Activate the virtual environment with:"
	@echo "  source $(VENV_DIR)/bin/activate"

## Install / refresh dependencies in existing .venv
install:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "No $(VENV_DIR) found. Run 'make env' first."; \
		exit 1; \
	fi
	@echo ">>> Installing/updating dependencies from requirements.txt"
	@$(PIP) install --upgrade pip
	@if [ -f requirements.txt ]; then \
		$(PIP) install -r requirements.txt; \
	else \
		echo "No requirements.txt found. Nothing to install."; \
	fi

## Install project plus development/test dependencies into existing .venv
dev:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "No $(VENV_DIR) found. Run 'make env' first."; \
		exit 1; \
	fi
	@echo ">>> Installing project with development/test dependencies"
	@$(PIP) install -e ".[dev]"

## Run the quick smoke-test suite
test:
	@if [ -x "$(VENV_PYTHON)" ]; then \
		$(VENV_PYTHON) -m pytest; \
	else \
		$(PYTHON) -m pytest; \
	fi

## Light clean: keep .venv, remove caches/build and notebook junk
clean:
	@echo ">>> Cleaning cache and build artifacts (keeping $(VENV_DIR))"
	rm -rf .pytest_cache dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf .virtual_documents
