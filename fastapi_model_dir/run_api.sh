#!/bin/bash

# Build Docker image from Dockerfile
docker build -t fastapi-model .

# Run Docker container
docker run -d -p 1313:1313 fastapi-model