from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backinthessr.trainer import Trainer
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
    data = {'Phrase' : text}
    X_pred = pd.DataFrame(data=data, index=[0])
    #model = joblib.load('model.joblib')
    model = tf.keras.models.load_model('model.h5')
    X_pred.transform(vocubal)
    print(X_pred)
    result = model.predict(X_pred)
    return { 'result' : result }
