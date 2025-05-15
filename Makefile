.PHONY: build run shell stop rm logs

IMAGE_NAME=auto-sale
CONTAINER_NAME=auto-sale

build:
	docker build -t $(IMAGE_NAME) -f docker/Dockerfile .

up:
	docker run -d --name $(CONTAINER_NAME) --restart=always -p 8000:8000 $(IMAGE_NAME)

down:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

logs:
	docker logs -f $(CONTAINER_NAME)
