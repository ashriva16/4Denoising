# Makefile: simple .venv workflow for users (not developers)

VENV_DIR := .venv
PYTHON   ?= python3
PIP      := $(VENV_DIR)/bin/pip

.PHONY: help env install clean

## Show available commands for users
help:
	@echo "Commands for using this project:"
	@echo "  make env       - create $(VENV_DIR) and install dependencies"
	@echo "  make install   - reinstall/update dependencies into existing $(VENV_DIR)"
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

## Light clean: keep .venv, remove caches/build and notebook junk
clean:
	@echo ">>> Cleaning cache and build artifacts (keeping $(VENV_DIR))"
	rm -rf .pytest_cache dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf .virtual_documents
