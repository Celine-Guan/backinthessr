FROM python:3.8.6-buster

COPY api /api
COPY backinthessr /backinthessr
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt
RUN trainer

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT