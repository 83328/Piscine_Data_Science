SHELL := /bin/bash
CONTAINER := piscineds_postgres
POSTGRES_USER := $(shell grep POSTGRES_USER ../.env | cut -d '=' -f2)
POSTGRES_DB := $(shell grep POSTGRES_DB ../.env | cut -d '=' -f2)

.PHONY: up down psql status clean prepare setup_db ex00 ex01 ex02

up:
	docker compose up -d

down:
	docker compose down -v

clean:
	@docker ps -a --filter "name=piscineds_" --format "{{.ID}}" | xargs -r docker rm -f

prepare:
	docker compose down --volumes --remove-orphans
	docker volume inspect piscineds_data >/dev/null 2>&1 && docker volume rm piscineds_data || true

logs:
	docker compose logs -f

psql:
	docker exec -it $(CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

status:
	docker ps -a --filter "name=piscineds" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

ex00:
	@python3 ex00/pie.py

ex01:
	@python3 ex01/chart.py
	
ex02:
	@python3 ex02/mustache.py