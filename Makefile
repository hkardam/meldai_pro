.PHONY: help build up down restart logs bash debug run-pipeline test export-postman download-ontologies clean

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
	@echo "  make logs           : Stream live logs from all containers"
	@echo "  make down           : Stop all running containers"
	@echo "  make clean          : Remove containers, volumes, and temporary caches"
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

logs-ui:
	docker compose logs -f frontend

bash:
	docker compose run --rm --service-ports app bash

debug:
	WAIT_FOR_DEBUGGER=true docker compose run --rm --service-ports app python -m meldai.main run-demo

run-pipeline:
	docker compose run --rm app python -m meldai.main run-demo

test:
	docker compose run --rm app pytest -v

export-postman:
	docker compose exec app python scripts/export_postman.py

download-ontologies:
	docker compose run --rm app python scripts/download_ontologies.py

clean:
	docker compose down -v
	rm -rf cache logs
