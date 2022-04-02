from fastapi import FastAPI, Request
from model import Model
import os
from pydantic import BaseModel

# Creating class to read from POST API
class Sentence(BaseModel):
    review: str
    

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the model
model = Model("model_weights/model.h5")

# Create the Fastapi app
app = FastAPI()

# Create the base endpoint
@app.get("/")
def hello():
    return {"result": "Applied Deep Learning Sample Deployment"}

# Create the prediction API
@app.post("/predict/")
async def predicts(sentence: Sentence):
    print(sentence.review)
    try:
        output = model.prediction(sentence.review)
        return {"rating": str(output[0])}
    except:
        return {"rating": "-1"}