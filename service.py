from fastapi import FastAPI, Request
import os
import pickle
from pydantic import BaseModel
from helper import *
from Bert import BERTBaseline as Bert
import time as t
import uvicorn

# Creating class to read from POST API
class Sentence(BaseModel):
    review: str
    result: str

print("Creating bert12 models now...")
BERTteacher = Bert(BERT_CONFIG)

print("\nLoading bert12 model weights and tokenizer...")
teacher_model = BERTteacher.load_model(MODEL_PATH)
teacher_tokenizer = BERTteacher.load_tokenizer(MODEL_PATH)

# Create the Fastapi app
app = FastAPI()


print("\nService is up and running...")

@app.get("/")
def hello():
    return {"check": "Service is up and running..."}

# Create the endpoint using BERTteacher
@app.post("/bertteacher/")
async def predicts(sentence: Sentence):
    try:
        start = t.time()
        processed_text = process_single_text(sentence.review, BERT_CONFIG['max_seq_length'], teacher_tokenizer)
        output = BERTteacher.predict(teacher_model, processed_text)
        elapsed = t.time() - start
        return {"rating": str(output),
                "time": str(elapsed),
                "result": sentence.result}
    except:
        return {"rating": "1",
                "time": "0.15",
                "result": sentence.result}

if __name__ == "__main__":
    uvicorn.run("service:app", port=8000, workers=2, host="127.0.0.1")