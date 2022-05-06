import pandas as pd
from transformers import BertTokenizer

from helper import *
from Bert import BERTBaseline as Bert

if __name__ == '__main__':

    set_seed(42)

    # print("\nTraining BERT Baseline")
    # logger = Logger(BERT_CONFIG)

    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # data = load_data("./Data/train.csv", BERT_CONFIG['max_seq_length'], bert_tokenizer)

    # trainer = Bert(BERT_CONFIG)
    # trainer.train(data, bert_tokenizer, MODEL_PATH)


    print("\n\n\nPrediction...")
    # Load the model and tokenizer
    predicter = Bert(BERT_CONFIG)
    model = predicter.load_model(MODEL_PATH)
    tokenizer = predicter.load_tokenizer(MODEL_PATH)
    print("Loaded model and tokenizer")

    # Process the text and get the prediction
    text = "Screen quality is worthless"
    processed_text = process_single_text(text, BERT_CONFIG['max_seq_length'], tokenizer)

    # Get the prediction
    prediction = predicter.predict(model, processed_text)
    print("Prediction:", prediction)

