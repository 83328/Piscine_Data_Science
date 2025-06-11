SHELL := /bin/bash
CONTAINER := piscineds_postgres
POSTGRES_USER := $(shell grep POSTGRES_USER ../.env | cut -d '=' -f2)
POSTGRES_DB := $(shell grep POSTGRES_DB ../.env | cut -d '=' -f2)

.PHONY: up down psql fclean clean ex01 ex02 ex03 start_db

clean:
	@echo "Stopping and removing all piscineds containers..."
	-docker rm -f piscineds_postgres piscineds_pgadmin 2>/dev/null || true
	-docker volume rm piscineds_data 2>/dev/null || true

up: clean
	docker compose up -d

down:
	docker compose down

fclean:
	docker compose down -v --remove-orphans
	docker rm -f $(CONTAINER) 2>/dev/null || true
	docker volume rm piscineds_data 2>/dev/null || true

psql:
	docker exec -it $(CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

ex01:
	@time cat ex01/customers_table.sql | docker exec -i $(CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

ex02:
	@time cat ex02/remove_duplicates.sql | docker exec -i $(CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

ex03:
	@echo "Running full fusion.sql setup..."
	docker exec -i $(CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) < ex03/fusion.sql

start_db:
	@echo "Preparing database..."
	@time cat ex01/customers_table.sql | docker exec -i $(CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)
	@time cat ex02/remove_duplicates.sql | docker exec -i $(CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)
	docker exec -i $(CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) < ex03/fusion.sql
	@echo "Database prepared successfully."
