# Makefile for Physics-Informed D2NN SciML Framework
# Automates local execution, Docker builds, training workflows, and evaluation.

PYTHON ?= python3
DOCKER ?= docker
IMAGE_NAME ?= d2nn-sciml:latest
UID := $(shell id -u)
GID := $(shell id -g)

.PHONY: help docker-build docker-compose-up test docker-test train-model-a train-model-b train-model-c eval-all clean

help:
	@echo "Available commands in Physics-Informed D2NN Makefile:"
	@echo "  make docker-build    : Build Docker container image ($(IMAGE_NAME))"
	@echo "  make test            : Run full unit test suite (pytest -v)"
	@echo "  make docker-test     : Run unit tests inside Docker container"
	@echo "  make train-model-a   : Train Model A (Phase-Only Baseline, seed 42)"
	@echo "  make train-model-b   : Train Model B (Complex-Valued Modulation)"
	@echo "  make train-model-c   : Train Model C (Apertured Phase Cascade)"
	@echo "  make eval-all        : Batch evaluate latest checkpoints for Models A, B, C"
	@echo "  make docker-compose-up: Launch training via Docker Compose with host outputs mounted"
	@echo "  make clean           : Clean Python cache, pytest cache, and temporary files"

docker-build:
	$(DOCKER) build -t $(IMAGE_NAME) .

docker-test:
	$(DOCKER) run --rm --entrypoint pytest $(IMAGE_NAME) -v

test:
	$(PYTHON) -m pytest -v

train-model-a:
	$(PYTHON) run_experiment.py \
		--config configs/default.yaml \
		--model-type phase_only \
		--tag model_a_phase_only \
		--seed 42

train-model-b:
	$(PYTHON) run_experiment.py \
		--config configs/complex_modulation.yaml \
		--model-type complex \
		--tag model_b_complex \
		--seed 42

train-model-c:
	$(PYTHON) run_experiment.py \
		--config configs/apertured_phase.yaml \
		--model-type apertured \
		--tag model_c_apertured_refined \
		--seed 42

eval-all:
	$(PYTHON) scripts/eval_all.py

docker-compose-up:
	UID=$(UID) GID=$(GID) docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
