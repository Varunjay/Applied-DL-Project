import pandas as pd
from helper import *
from LSTMBaseline import LSTMBaseline as LSTM

if __name__ == '__main__':

    set_seed(42)

    print("\nTraining BiLSTM Baseline")
    logger = Logger(BERT_CONFIG)


    data = pd.read_csv("./Data/train.csv")
    X_train = data['review'].values
    y_train = data['rating'].values

    label_dic = {label: i for i, label in enumerate(set([rating for rating in y_train]))}

    y_train = data['rating'].apply(lambda x: label_dic[x]).values

    trainer = LSTM(BILSTM_CONFIG)

    trainer.train(X_train, y_train, y_train, MODEL_PATH)
    
    print("\nModel Training done...")
    
    # print("\n\n\nPrediction...")
    # # Load the model and tokenizer
    # predicter = Bert(BERT_CONFIG)
    # model = predicter.load_model(MODEL_PATH)
    # tokenizer = predicter.load_tokenizer(MODEL_PATH)
    # print("Loaded model and tokenizer")

    # # Process the text and get the prediction
    # text = "Its worth it for its price. But given another chance I wouldn't buy it."
    # processed_text = process_single_text(text, BERT_CONFIG['max_seq_length'], tokenizer)

    # # Get the prediction
    # prediction = predicter.predict(model, processed_text)
    # print("Prediction:", prediction)

