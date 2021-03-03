# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* backinthessr/*.py

black:
	@black scripts/* backinthessr/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr backinthessr-*.dist-info
	@rm -fr backinthessr.egg-info

install:
	@pip install . -U

all: clean install test black check_code


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      		RUNNING API
# ----------------------------------
run_api:
	uvicorn api.fast:app --reload 

# ----------------------------------
#      	DOCKER DEPLOYMENT
# ----------------------------------

GCP_PROJECT_ID=wagon-corentin-305109
DOCKER_IMAGE_NAME=backinthessr-test-01
GCR_MULTI_REGION=eu.gcr.io
GCR_REGION=europe-west1

docker_build:
	docker build --network=host -t ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} .

docker_run:
	docker run -e PORT=8000 -p 8000:8000 ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}

docker_push:
	docker push ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}

gcloud_run:
	gcloud run deploy --image ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region ${GCR_REGION}
