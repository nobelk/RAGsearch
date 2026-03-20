IMAGE ?= registry:5000/app:latest
REMOTE_HOST ?= deploy-host
REMOTE_DIR ?= ~/app

# -- Build & Dependencies --
install:        ## Install all deps + build UI
	uv sync
	cd app_ui && flutter pub get && flutter build web

# -- Run --
run:            ## Run the API server
	uv run app

# -- Test --
test:           ## Unit tests only
	uv run pytest -m "not integration"
test-all:       ## All tests including integration
	uv run pytest
test-file:      ## Run a single test file (usage: make test-file FILE=tests/llm_tests.py)
	uv run pytest $(FILE)

# -- Formatting --
format:         ## Format code with black
	uv run black src/ tests/
format-check:   ## Check formatting without modifying
	uv run black --check src/ tests/

# -- Services --
services-up:    ## Start Qdrant + Ollama
	docker compose up -d qdrant ollama
services-down:  ## Stop all services
	docker compose down
qdrant-up:      ## Start Qdrant only
	docker compose up -d qdrant
qdrant-down:    ## Stop Qdrant only
	docker compose stop qdrant

# -- UI --
ui-build:       ## Build the Flutter web UI
	cd app_ui && flutter pub get && flutter build web
ui-run:         ## Build and serve the Flutter web UI in Chrome
	cd app_ui && flutter run -d chrome --web-port=3000

# -- Ingest --
ingest:         ## Ingest PDFs into Qdrant
	uv run app-ingest data/

# -- Deploy --
deploy: deploy-build deploy-push deploy-start  ## Full deploy: build UI + image, push, start

deploy-build:   ## Build Flutter UI + Docker image
	cd app_ui && flutter build web --dart-define=API_BASE_URL= --release
	rm -rf static && cp -r app_ui/build/web static
	docker buildx build --platform linux/arm64 -t $(IMAGE) -f Dockerfile --load .

deploy-push:    ## Push image to registry
	docker push $(IMAGE)

deploy-start:   ## Transfer compose + restart services on remote host
	scp deploy/docker-compose.deploy.yml $(REMOTE_HOST):$(REMOTE_DIR)/docker-compose.yml
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && docker compose pull && docker compose up -d"
	@echo "Waiting for health check..."
	@for i in 1 2 3 4 5 6; do \
		sleep 5; \
		if ssh $(REMOTE_HOST) "curl -sf http://localhost:8080/health" > /dev/null 2>&1; then \
			echo "Service is healthy!"; exit 0; \
		fi; \
		echo "Attempt $$i: not ready yet..."; \
	done; \
	echo "WARNING: health check did not pass within 30s"

deploy-setup: deploy  ## First-time: deploy + transfer data + pull models + ingest
	scp -r data/ $(REMOTE_HOST):$(REMOTE_DIR)/data/
	$(MAKE) deploy-models
	$(MAKE) deploy-ingest

deploy-models:  ## Pull Ollama models on remote host
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && docker compose exec ollama ollama pull nomic-embed-text"
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && docker compose exec ollama ollama pull llama3.2"
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && docker compose exec ollama ollama pull llama3.2:1b"

deploy-ingest:  ## Run ingestion on remote host
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && docker compose exec app uv run app-ingest data/"

deploy-status:  ## Show remote container status
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && docker compose ps"

deploy-logs:    ## Tail remote logs
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && docker compose logs -f --tail=50"

deploy-stop:    ## Stop remote services
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && docker compose down"

.PHONY: install run test test-all test-file format format-check \
        services-up services-down qdrant-up qdrant-down \
        ui-build ui-run ingest \
        deploy deploy-build deploy-push deploy-start deploy-setup \
        deploy-models deploy-ingest deploy-status deploy-logs deploy-stop
