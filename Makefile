.PHONY: help build up down restart logs bash debug run-pipeline test export-postman download-ontologies analyze accuracy-matrix molecule-confusion-matrix lighten-json analyze-all clean

help:
	@echo "======================================================================"
	@echo "🏥 MeldAI Medical Science Platform - Development Commands"
	@echo "======================================================================"
	@echo "  make up             : Start PostgreSQL, MongoDB, and MeldAI App"
	@echo "  make debug          : Run MeldAI container in interactive debug mode"
	@echo "  make bash           : Open an interactive bash shell in the app container"
	@echo "  make run-pipeline         : Execute end-to-end PG -> SapBERT -> Mongo"
	@echo "  make download-ontologies  : Download HPO (hp.obo) and MONDO (mondo.obo) OBO files"
	@echo "  make test           : Run automated test suite inside Docker"
	@echo "  make export-postman : Generate/sync Postman Collection JSON"
	@echo "  make analyze               : Run prescription accuracy matrix & charts (FILE=path.json)"
	@echo "  make molecule-confusion-matrix : Per-molecule OvR confusion matrix (FILE=, TOP=40, PLOT=1)"
	@echo "  make lighten-json           : Strip prompts from result JSON into audit file (FILE=, DRY=0)"
	@echo "  make analyze-all            : Run accuracy-matrix + molecule-confusion-matrix"
	@echo "  make logs                   : Stream live logs from all containers"
	@echo "  make down                   : Stop all running containers"
	@echo "  make clean                  : Remove containers, volumes, and temporary caches"
	@echo "======================================================================"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

logs-engine:
	docker compose logs -f app

bash:
	docker compose run --rm --service-ports app bash

debug:
	DEBUG_PORT=5679 WAIT_FOR_DEBUGGER=true docker compose run --rm --service-ports app python -m main serve --host 0.0.0.0 --port 8000

run-pipeline:
	docker compose run --rm app python -m main run-demo

test:
	docker compose run --rm app pytest -v

export-postman:
	docker compose exec app python scripts/export_postman.py

download-ontologies:
	docker compose run --rm app python scripts/download_ontologies.py

analyze:
	@if docker compose ps --services --filter "status=running" | grep -q "^app$$"; then \
		docker compose exec app python scripts/analyzer/accuracy_matrix.py $(FILE); \
	else \
		docker compose run --rm -e DEBUG_MODE=false app python scripts/analyzer/accuracy_matrix.py $(FILE); \
	fi

accuracy-matrix: analyze

# TOP=40  – number of top items to display/chart  (default 40)
# PLOT=1  – set to 0 to skip PNG chart generation (default 1)
molecule-confusion-matrix:
	@_EXTRA="$(if $(FILE),--file $(FILE),) --top $(or $(TOP),40) $(if $(filter 0,$(PLOT)),--no-plot,)"; \
	if docker compose ps --services --filter "status=running" | grep -q "^app$$"; then \
		docker compose exec app python3 scripts/analyzer/molecule_confusion_matrix.py $$_EXTRA; \
	else \
		docker compose run --rm -e DEBUG_MODE=false app python3 scripts/analyzer/molecule_confusion_matrix.py $$_EXTRA; \
	fi

# DRY=1   – print size report only, do not write output files (default 0)
lighten-json:
	@_EXTRA="$(if $(FILE),--file $(FILE),) $(if $(filter 1,$(DRY)),--dry-run,)"; \
	if docker compose ps --services --filter "status=running" | grep -q "^app$$"; then \
		docker compose exec app python3 scripts/analyzer/lighten_result_json.py $$_EXTRA; \
	else \
		docker compose run --rm -e DEBUG_MODE=false app python3 scripts/analyzer/lighten_result_json.py $$_EXTRA; \
	fi

analyze-all: analyze molecule-confusion-matrix

clean:
	docker compose down -v
	rm -rf cache logs
