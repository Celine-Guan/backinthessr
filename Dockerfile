FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet

COPY api /api
COPY backinthessr /backinthessr
COPY model.h5 /model.h5
COPY vocabulary.joblib /vocabulary.joblib

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT