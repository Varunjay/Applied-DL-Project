import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from numpy import argmax
import tensorflow_text as text
from sklearn.metrics import confusion_matrix, classification_report
import sklearn.metrics as skm
import time

class ModelTraining:
    def __init__(self, name = "", layer = "12_H-768_A-12/"):

        self.model_name = "model" + name
        self.layer = layer

    def creat_model(self):
        bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-" + self.layer)
        # Bert layers
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)

        # Neural network layers
        l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
        l = tf.keras.layers.Dense(3, activation='softmax', name="output")(l)


        # Use inputs and outputs to construct a final model
        self.model = tf.keras.Model(inputs=[text_input], outputs = [l])
    
    def get_model_summary(self):
        print(self.model.summary())
    
    def compile_model(self):
        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]

        self.model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=METRICS)
    
    def train_model(self, X_train, y_train, epochs = 10):
        self.model.fit(X_train, np.asarray(y_train), epochs = epochs)

    def save_mode(self, data_path = ""):
        self.model.save(data_path + self.model_name)
    
    def load_model(self, data_path = ""):
        self.model = tf.keras.models.load_model(data_path + self.model_name)
    
    def save_model_weights(self, data_path = ""):
        self.model.save_weights(data_path + self.model_name + "_weights.h5")
    
    def load_model_weights(self, data_path = ""):
        self.model.load_weights(data_path + self.model_name + "_weights.h5")

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        print(skm.classification_report(y_test, y_pred))
        print(skm.confusion_matrix(y_test, y_pred))
    
    def predict_model(self, X):
        start = time()
        if type(X) != list:
            X = [X]
        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        elapsed = time() - start
        return y_pred, elapsed

    def get_number_of_total_parameters(self):
        return self.model.count_params()

    def get_number_of_trainable_parameters(self):
        return sum([np.prod(param.get_shape().as_list()) for param in self.model.trainable_variables()])

    def get_number_of_non_trainable_parameters(self):
        return sum([np.prod(param.get_shape().as_list()) for param in self.model.non_trainable_variables()])

if "__name__" == "main":
    model = ModelTraining("bert12", 3)