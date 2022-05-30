help:
	cat Makefile
run:
	docker-compose up
exec:
	docker-compose run mo445 bash
build: stop .FORCE
	docker-compose build
rebuild: stop .FORCE
	docker-compose build --force-rm
stop:
	docker stop mo445 || true; docker rm mo445 || true;
.FORCE: