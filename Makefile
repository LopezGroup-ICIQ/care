# Define the image name and tag
IMAGE_NAME := care
TAG := latest

.PHONY: build run clean build-and-run debug status

# Default target, build the Docker image only if it doesn't exist
build:
	docker build -t $(IMAGE_NAME):$(TAG) .;

# Run the container (only build if the image is not present)
run:
	docker compose up --remove-orphans;

# Clean up (optional, removes the image)
clean:
	docker compose down; \
	docker rmi -f $(IMAGE_NAME):$(TAG);

# Debug into the container
debug:
	docker compose run --entrypoint /bin/sh care;

# Show the status of the container
status:
	docker ps -a | grep $(IMAGE_NAME);

# For convenience, you can define a target to build and run together
build-and-run: build run