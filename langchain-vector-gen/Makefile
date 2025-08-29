SHELL=/bin/bash

source/venv/bin:
	. ./venv/bin/activate

install-prod: source/venv/bin
	pip install -r requirements.txt

install-dev: source/venv/bin install-prod
	pip install -r requirements-dev.txt

build:
	python -m generator.load_vector_db

run:
	 python -m generator.main