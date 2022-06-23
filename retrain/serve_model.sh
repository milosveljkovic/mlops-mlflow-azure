#!/bin/sh

docker build mlflow_docker:latest . & docker run -p 1234:1234 --network=host mlflow_docker:latest