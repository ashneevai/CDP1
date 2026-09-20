.PHONY: setup test test-unit test-integration test-golden test-performance \
	architecture lint quality run down logs clean clean-runtime-data evaluation

setup:
	@if [ ! -f .env ]; then cp .env.example .env; echo "created .env from .env.example"; fi
	python -m pip install -e ".[dev]"

test: test-unit

test-unit:
	@mkdir -p .test-tmp
	python -m pytest tests/unit tests/architecture -q -p no:cacheprovider --basetemp=.test-tmp/pytest

test-golden:
	@mkdir -p .test-tmp
	python -m pytest tests/golden -q -m golden -p no:cacheprovider --basetemp=.test-tmp/golden

test-integration:
	docker compose up -d --wait
	@mkdir -p .test-tmp
	python -m pytest tests/integration -q -m integration -p no:cacheprovider --basetemp=.test-tmp/integration
	docker compose down

test-performance:
	@mkdir -p .test-tmp
	python -m pytest tests/performance -q -m performance -p no:cacheprovider --basetemp=.test-tmp/performance

architecture:
	python scripts/check_architecture.py

lint:
	ruff check apps packages workers evaluation scripts tests

quality: architecture lint test-unit

run:
	docker compose up -d --build --wait
	@echo "Ingestion API:    http://localhost:8000/docs"
	@echo "Human review UI:  http://localhost:8100/ui/review-tasks"
	@echo "MinIO console:    http://localhost:9001  (minioadmin / minioadmin)"
	@echo "Redpanda admin:   http://localhost:9644"

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	python scripts/clean_workspace.py

# Destructive by design: unlike `clean`, this also removes local service data.
clean-runtime-data:
	docker compose down -v

evaluation:
	python -m evaluation.runner --dataset dataset_raw --ground-truth evaluation_data/ground_truth.json --predictions evaluation_data/predictions.json --output evaluation_results
