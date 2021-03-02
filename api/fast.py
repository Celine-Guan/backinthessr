from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backinthessr.trainer import Trainer
from backinthessr.clean_data2 import remove_punctuation, lowercase, lemma, split_sentences

import joblib
import pandas as pd
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict_sentiment(text):
    # Get X_pred
    text = remove_punctuation(text)
    text = lowercase(text)
    text= lemma(text)
    data = {'Phrase': text}
    X_pred = pd.Series(data)
    X_pred = split_sentences(X_pred)

    # Get model & vocabulary
    model = tf.keras.models.load_model('model.h5')
    voc = joblib.load('vocabulary.joblib')

    # Transform X_pred
    X_pred_pad = voc.transform(X_pred)

    result = model.predict(X_pred_pad)
    return { 'result' : float(result[0][0]) }
