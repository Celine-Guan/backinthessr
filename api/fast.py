from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backinthessr.trainer import Trainer

app = FastAPI()

model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict_sentiment():
    data = {'Phrase' : text}
    X_pred = pd.DataFrame(data=data)

    y = train
    result = model.predict(X_pred)
    return { 'result' : model.predict()}