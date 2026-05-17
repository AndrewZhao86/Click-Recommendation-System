.PHONY: up up-obs down down-v logs ps migrate seed api consumer replay replay-dlq loadtest loadtest-preflight loadtest-llm loadtest-sweep check-loadtest eval-llm eval-offline test fmt lint typecheck install docker-build docker-up docker-down docker-logs

N ?= 1
EVENTS ?= 10000

install:
	uv sync --extra dev

up:
	docker compose up -d
	@echo "Waiting for services..."
	@docker compose ps

# Phase 8b: opt-in profile that adds Jaeger to the stack.
up-obs:
	docker compose --profile observability up -d
	@docker compose --profile observability ps

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

# Sanity-check that the API is up AND that seed/replay populated the
# ranker's dependencies. Without this guard a loadtest run can record a
# `/search`-503 artefact for the entire 5 min — which is exactly how
# the previous run produced 1724/1728 failures with p95=25s.
loadtest-preflight:
	@echo "preflight: checking /search returns 200 (seed + replay required)"
	@uv run python -c "import sys, urllib.request; \
	r = urllib.request.urlopen('http://localhost:8000/search?q=wireless+headphones&user_id=u_brand_loyalist_0001&limit=1', timeout=5); \
	sys.exit(0 if r.status == 200 else 1)" \
	  || (echo 'FAIL: /search did not return 200. Run: make seed && make replay EVENTS=20000 && restart `make api`.' && exit 1)

# Phase 8c headline run — 500 users × 5 min, headless, HTML + CSV report.
loadtest: loadtest-preflight
	mkdir -p artifacts
	uv run locust -f tests/load/locustfile.py --host http://localhost:8000 \
	  --users 500 --spawn-rate 50 --run-time 5m --headless \
	  --html artifacts/locust_report.html --csv artifacts/locust

# Phase 8c LLM-only fallback exercise — small + slow on purpose.
loadtest-llm:
	mkdir -p artifacts
	uv run locust -f tests/load/locustfile_llm.py --host http://localhost:8000 \
	  --users 10 --spawn-rate 2 --run-time 2m --headless \
	  --html artifacts/locust_llm_report.html

# Phase 8c scalability proof — N=1, 3, 6 consumers.
# Python port of the original bash script so the sweep runs on Windows
# hosts as well (the previous `pkill -f` relied on POSIX).
loadtest-sweep:
	uv run python scripts/throughput_sweep.py

# Phase 8c acceptance check — assert p95, failure count, scaling ratio,
# cache hit ratio and consumer lag. The metrics scrape is on by default
# so a stale artefact on disk can never quietly pass; override the URL
# (or pass `METRICS_URL=` to disable it) when running against a
# different host.
METRICS_URL ?= http://localhost:8000
check-loadtest:
	uv run python scripts/check_loadtest_results.py \
	  $(if $(METRICS_URL),--metrics-url $(METRICS_URL)) $(ARGS)

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

# ---- Phase 9: containerised app ----

docker-build:
	docker build -t click-rec:local .

docker-up:
	docker compose --profile app up -d
	@docker compose --profile app ps

docker-down:
	docker compose --profile app down

docker-logs:
	docker compose --profile app logs -f app

# Trivy CVE scanning runs in CI (see .github/workflows/ci.yml). Not exposed
# as a local target — Linux CI runners scan the 3 GB image in ~2 min, whereas
# Windows hits PowerShell pipe + memory issues on the same workload.
