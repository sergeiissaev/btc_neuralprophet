#!/usr/bin/env bash
PROJECT=btc_prediction_repo

build up start stop down push pull :
	DOCKER_BUILDKIT=1 docker-compose $@

deploy:
	DOCKER_BUILDKIT=1 docker-compose up -d --build

docker_build:
	DOCKER_BUILDKIT=1 docker build . -t $(PROJECT)

docker_run:
	docker run -it --network host -v $(shell pwd):/opt/btc_prediction_repo $(PROJECT)

docker_bash:
	docker run -it --network host -v $(shell pwd):/opt/btc_prediction_repo $(PROJECT) bash

docker_test:
	docker run -it --network host -v $(shell pwd):/opt/btc_prediction_repo $(PROJECT) pytest --cov-report term-missing --cov=btc_prediction_repo tests/


