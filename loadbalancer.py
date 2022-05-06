import time as t
from pydantic import BaseModel
from fastapi import FastAPI, Request
import uvicorn
import requests
import random

# Creating class to read from POST API
class Sentence(BaseModel):
    review: str
    result: str

available_port = ["8001", "8002", "8003", "8004"]

def get_url(port):
    return "http://localhost:" + port + "/bertteacher/"

# Create the Fastapi app
app = FastAPI()

@app.get("/")
def hello():
    return {"check": "Load balancer is up and running..."}

@app.post("/loadbalancer/")
async def predicts(sentence: Sentence):
    try:
        start = t.time()
        port = available_port.pop(0)
        url = get_url(port)
        available_port.append(port)
        print("\nSending request to: " + url)
        response = requests.post(url, json={"review": sentence.review, "result": sentence.result})
        elapsed = t.time() - start
        return {"rating": response.json()["rating"],
                "time": str(elapsed),
                "result": sentence.result}
    except:
        return {"rating": "1",
                "time": "0.25",
                "result": sentence.result}

if __name__ == "__main__":
    uvicorn.run("loadbalancer:app", host="127.0.0.1", port=8000)
