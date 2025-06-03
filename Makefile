# Makefile


CLIENT_IMAGE_NAME := minhbui1/client-infer
CLIENT_IMAGE_TAG := v2.0
SERVER_IMAGE_NAME := minhbui1/server-infer
SERVER_IMAGE_TAG := v2.0


DOCKERFILE_CLIENT := Dockerfile.client
DOCKERFILE_SERVER := Dockerfile.server

COMPOSE_FILE := deployments/docker/docker-compose.yaml

# ==============================================================================
# Docker Build Targets
# ==============================================================================

.PHONY: build-client
build-client:
	@echo "Building client Docker image: $(CLIENT_IMAGE_NAME):$(CLIENT_IMAGE_TAG)..."
	@docker build -t $(CLIENT_IMAGE_NAME):$(CLIENT_IMAGE_TAG) -f $(DOCKERFILE_CLIENT) .
	@echo "Client image built."

.PHONY: build-server
build-server:
	@echo "Building server Docker image: $(SERVER_IMAGE_NAME):$(SERVER_IMAGE_TAG)..."
	@docker build -t $(SERVER_IMAGE_NAME):$(SERVER_IMAGE_TAG) -f $(DOCKERFILE_SERVER) .
	@echo "Server image built."

.PHONY: build-all
build-all: build-client build-server
	@echo "All Docker images built."


# ==============================================================================
# Docker Push Targets 
# ==============================================================================

.PHONY: push-client
push-client: build-client # Đảm bảo image đã được build trước khi push
	@echo "Pushing client Docker image: $(CLIENT_IMAGE_NAME):$(CLIENT_IMAGE_TAG) to Docker Hub..."
	@docker push $(CLIENT_IMAGE_NAME):$(CLIENT_IMAGE_TAG)
	@echo "Client image $(CLIENT_IMAGE_NAME):$(CLIENT_IMAGE_TAG) pushed."

.PHONY: push-server
push-server: build-server # Đảm bảo image đã được build trước khi push
	@echo "Pushing server Docker image: $(SERVER_IMAGE_NAME):$(SERVER_IMAGE_TAG) to Docker Hub..."
	@docker push $(SERVER_IMAGE_NAME):$(SERVER_IMAGE_TAG)
	@echo "Server image $(SERVER_IMAGE_NAME):$(SERVER_IMAGE_TAG) pushed."

.PHONY: push-all
push-all: push-client push-server
	@echo "All Docker images pushed to Docker Hub."


# ==============================================================================
# Docker Compose Targets
# ==============================================================================

.PHONY: start
start:
	@echo "Starting services with Docker Compose..."
	@docker compose -f $(COMPOSE_FILE) up -d
	@echo "Services started in detached mode."
	@echo "Run 'make logs' to view logs."

.PHONY: stop
stop:
	@echo "Stopping services with Docker Compose..."
	@docker compose -f $(COMPOSE_FILE) down
	@echo "Services stopped and removed."

.PHONY: restart
restart: stop start
	@echo "Services restarted."

# ==============================================================================
# Utility Targets
# ==============================================================================

.PHONY: logs
logs:
	@echo "Following logs from Docker Compose..."
	@docker compose -f $(COMPOSE_FILE) logs -f --tail="50"

.PHONY: ps
ps:
	@echo "Current Docker Compose services status:"
	@docker compose -f $(COMPOSE_FILE) ps

.PHONY: clean-volumes
clean-volumes: stop
	@echo "Removing Docker Compose volumes (rabbitmq_data, redis_data, redisinsight_data)..."
	@docker volume rm $$(docker volume ls -qf "name=rabbitmq_data") || true
	@docker volume rm $$(docker volume ls -qf "name=redis_data") || true
	@docker volume rm $$(docker volume ls -qf "name=redisinsight_data") || true
	@echo "Volumes cleaned."
	
.PHONY: prune
prune: stop
	@echo "Pruning unused Docker images, containers, volumes, and networks..."
	@docker system prune -af
	@echo "Docker system pruned."


# Default target (chạy khi gõ 'make' không có tham số)
.PHONY: default
default: build-all start

# Target trợ giúp
.PHONY: help
help:
	@echo "Available Makefile targets:"
	@echo "  build-client         - Build the client Docker image"
	@echo "  build-server         - Build the server Docker image"
	@echo "  build-all            - Build all Docker images (client and server)"
	@echo "  start                - Start all services defined in docker-compose.yaml in detached mode"
	@echo "  stop                 - Stop and remove all services defined in docker-compose.yaml"
	@echo "  restart              - Restart all services (stop then start)"
	@echo "  logs                 - Follow logs from all running services"
	@echo "  ps                   - List running Docker Compose services"
	@echo "  clean-volumes        - Stop services and remove associated Docker volumes"
	@echo "  prune                - Stop services and prune unused Docker resources"
	@echo "  help                 - Show this help message"
	@echo ""
	@echo "You can override image names and tags:"
	@echo "  make build-client CLIENT_IMAGE_NAME=mycustom/client CLIENT_IMAGE_TAG=latest"