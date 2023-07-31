test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	docker build --rm -t pistachio-classifier-service .

integration_test: build
	LOCAL_IMAGE_NAME=pistachio-classifier-service bash integraton-test/run.sh


setup:
	pipenv install --dev
	pre-commit install