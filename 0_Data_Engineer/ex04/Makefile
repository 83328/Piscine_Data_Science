.PHONY: up down logs psql status sql clean automate network items

# Define environment variables
POSTGRES_USER := alimpens
POSTGRES_DB := piscineds

up: clean prepare
	docker compose up -d

prepare:
	docker compose down --volumes --remove-orphans
	docker volume inspect piscineds_data >/dev/null 2>&1 && docker volume rm piscineds_data || true

down:
	docker compose down -v

clean:
	@docker ps -a --filter "name=piscineds_" --format "{{.ID}}" | xargs -r docker rm -f

logs:
	docker compose logs -f

psql:
	docker exec -it piscineds_postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

status:
	docker ps -a --filter "name=piscineds_postgres" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

sql:
	cat table.sql | docker exec -i piscineds_postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

items:
	python3 items_table.py