SHELL := /usr/bin/env bash

PYTHON ?= python3
API_HOST ?= 127.0.0.1
API_PORT ?= 8788
UI_HOST ?= 127.0.0.1
UI_PORT ?= 8502

.PHONY: install run-api run-ui test check format docker-api docker-dashboard

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

run-api:
	$(PYTHON) -m uvicorn api.main:app --reload --host $(API_HOST) --port $(API_PORT)

run-ui:
	$(PYTHON) -m streamlit run dashboard/app.py --server.address $(UI_HOST) --server.port $(UI_PORT)

test:
	$(PYTHON) -m pytest -q

check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy agents api dashboard ml simulator
	$(PYTHON) -m pytest -q

format:
	$(PYTHON) -m ruff format .

docker-api:
	docker compose up --build

docker-dashboard:
	docker compose --profile dashboard up --build
