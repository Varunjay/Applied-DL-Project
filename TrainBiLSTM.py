import pandas as pd
from helper import *
from LSTMBaseline import LSTMBaseline as LSTM

if __name__ == '__main__':

    set_seed(42)

    # print("\nTraining BiLSTM Baseline")

    # data = pd.read_csv("./Data/train.csv")
    # X_train = data['review'].values
    # y_train = data['rating'].values

    # label_dic = {label: i for i, label in enumerate(set([rating for rating in y_train]))}

    # y_train = data['rating'].apply(lambda x: label_dic[x]).values

    # trainer = LSTM(BILSTM_CONFIG)

    # trainer.train(X_train, y_train, y_train, MODEL_PATH)

    print("\n\n\nPrediction...")
    # Load the model and tokenizer
    predicter = LSTM(BILSTM_CONFIG)

    text_field = predicter.load_vocab(MODEL_PATH)
    model = predicter.load_model(MODEL_PATH, text_field)

    # Process the text and get the prediction
    text = "Such poor quality. I am not happy with this product."

    # Get the prediction
    prediction = predicter.predict(model, text_field, text)
    print("Prediction:", prediction)

    text = "wow, this screen pixels is 4k, so I can see the whole world."

    # Get the prediction
    prediction = predicter.predict(model, text_field, text)
    print("Prediction:", prediction)

