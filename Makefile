.PHONY: up down down-v logs ps migrate seed api consumer replay replay-dlq loadtest eval-llm eval-offline test fmt lint typecheck install

N ?= 1
EVENTS ?= 10000

install:
	uv sync --extra dev

up:
	docker compose up -d
	@echo "Waiting for services..."
	@docker compose ps

down:
	docker compose down

down-v:
	docker compose down -v

logs:
	docker compose logs -f

ps:
	docker compose ps

migrate:
	uv run alembic upgrade head

seed:
	uv run python -m click_rec.cli seed

api:
	uv run uvicorn click_rec.api.app:app --host 0.0.0.0 --port 8000 --reload

consumer:
	uv run python -m click_rec.cli consumer --workers $(N)

replay:
	uv run python -m click_rec.cli replay --events $(EVENTS)

replay-dlq:
	uv run python -m click_rec.cli replay-dlq

loadtest:
	uv run locust -f tests/load/locustfile.py --host http://localhost:8000

eval-llm:
	uv run python -m click_rec.cli eval-llm $(ARGS)

eval-offline:
	uv run python -m click_rec.cli eval-offline $(ARGS)

test:
	uv run pytest -v

fmt:
	uv run ruff format src tests

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src
