.PHONY: build run

build:
	ollama create cv-model -f Modelfile

run:
	cat query.txt | ollama run cv-model