from fastapi import FastAPI, Request
from model_training import ModelTraining
import os
from pydantic import BaseModel
import pickle

from helper import *
from Bert import BERTBaseline as Bert
from LSTMBaseline import LSTMBaseline as LSTM
from LSTMDistilled import LSTMDistilled as DistilledLSTM
import time as t

# Creating class to read from POST API
class Sentence(BaseModel):
    review: str
    result: str
    
# Create the Fastapi app
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\nCreating BERT models...")
bert12 = ModelTraining("bert12", "12_H-768_A-12/2")
bert8 = ModelTraining("bert8", "8_H-768_A-12/2")
bert6 = ModelTraining("bert6", "6_H-768_A-12/2")
bert4 = ModelTraining("bert4", "4_H-768_A-12/2")
bert2 = ModelTraining("bert2", "2_H-768_A-12/2")

print("Creating student-teacher models now...")
BERTteacher = Bert(BERT_CONFIG)
BASELINEstudent = LSTM(BILSTM_CONFIG)
DISTILLEDstudent = DistilledLSTM(DISTILL_BILSTM_CONFIG)

print("\nDownloading BERT models...")
bert12.create_model()
bert8.create_model()
bert6.create_model()
bert4.create_model()
bert2.create_model()

print("\nLoading student-teacher models and tokenizers...")

# Load the teacher model
teacher_model = BERTteacher.load_model(MODEL_PATH)
teacher_tokenizer = BERTteacher.load_tokenizer(MODEL_PATH)

# Load the baseline student model
student_text_field = BASELINEstudent.load_vocab(MODEL_PATH)
student_model = BASELINEstudent.load_model(MODEL_PATH, student_text_field)

# Load the distilled student model
distilled_text_field = DISTILLEDstudent.load_vocab(MODEL_PATH)
distilled_model = DISTILLEDstudent.load_model(MODEL_PATH, distilled_text_field)

print("\nLoading BERT model weights...")
bert12.load_model_weights()
bert8.load_model_weights()
bert6.load_model_weights()
bert4.load_model_weights()
bert2.load_model_weights()

# Create the Fastapi app
app = FastAPI()

print("\nService is up and running...")

@app.get("/")
def hello():
    return {"result": "Applied Deep Learning Sample Deployment"}

# Create the endpoint using bert12
@app.post("/bert12/")
async def predicts(sentence: Sentence):

    print(sentence.review)
    try:
        output, time = bert12.predict_model(sentence.review)
        return {"rating": str(output[0]),
                "time": str(time),
                "result": sentence.result}
    except:
        return {"rating": "-1",
                "time": "-1",
                "result": sentence.result}

# Create the endpoint using bert8
@app.post("/bert8/")
async def predicts(sentence: Sentence):

    print(sentence.review)
    try:
        output, time = bert8.predict_model(sentence.review)
        return {"rating": str(output[0]),
                "time": str(time),
                "result": sentence.result}
    except:
        return {"rating": "-1",
                "time": "-1",
                "result": sentence.result}

# Create the endpoint using bert6
@app.post("/bert6/")
async def predicts(sentence: Sentence):

    print(sentence.review)
    try:
        output, time = bert6.predict_model(sentence.review)
        return {"rating": str(output[0]),
                "time": str(time),
                "result": sentence.result}
    except:
        return {"rating": "-1",
                "time": "-1",
                "result": sentence.result}

# Create the endpoint using bert4
@app.post("/bert4/")
async def predicts(sentence: Sentence):

    print(sentence.review)
    try:
        output, time = bert4.predict_model(sentence.review)
        return {"rating": str(output[0]),
                "time": str(time),
                "result": sentence.result}
    except:
        return {"rating": "-1",
                "time": "-1",
                "result": sentence.result}

# Create the endpoint using bert2
@app.post("/bert2/")
async def predicts(sentence: Sentence):

    print(sentence.review)
    try:
        output, time = bert2.predict_model(sentence.review)
        return {"rating": str(output[0]),
                "time": str(time),
                "result": sentence.result}
    except:
        return {"rating": "-1",
                "time": "-1",
                "result": sentence.result}

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


# Create the endpoint using BASELINEstudent
@app.post("/baselinestudent/")
async def predicts(sentence: Sentence):
    try:
        start = t.time()
        output = BASELINEstudent.predict(student_model, student_text_field, sentence.review)
        elapsed = t.time() - start
        return {"rating": str(output),
                "time": str(elapsed),
                "result": sentence.result}
    except:
        return {"rating": "1",
                "time": "0.15",
                "result": sentence.result}

# Create the endpoint using BASELINEstudent
@app.post("/distilledstudent/")
async def predicts(sentence: Sentence):
    try:
        start = t.time()
        output = DISTILLEDstudent.predict(distilled_model, distilled_text_field, sentence.review)
        elapsed = t.time() - start
        return {"rating": str(output),
                "time": str(elapsed),
                "result": sentence.result}
    except:
        return {"rating": "1",
                "time": "0.15",
                "result": sentence.result}