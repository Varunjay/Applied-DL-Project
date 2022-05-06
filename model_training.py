import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from numpy import argmax
import tensorflow_text as text
import time

class ModelTraining:
    def __init__(self, name = "", layer = "12_H-768_A-12/2"):
        """
        Initialize the ModelTraining class
        Args:
            name: Name of the model
            layer: Type of the BERT model
        """

        self.model_name = "model" + name
        self.layer = layer

    def create_model(self):
        """
        Create the model using the BERT model 
        """
        bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

        print("Getting model from: " + "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-" + self.layer)
        bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-" + self.layer)


        # Bert layers
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)

        # Neural network layers
        l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
        l = tf.keras.layers.Dense(3, activation='softmax', name="output")(l)


        # Use inputs and outputs to construct a final model
        self.model = tf.keras.Model(inputs=[text_input], outputs = [l])

        # Note: The model is not used in this example, but it is useful to keep
    def create_student(self):
        """
        Create the student model
        """

        print("Creating student model ----")
        bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='input')
        preprocessed_text = bert_preprocess(text_input)
        flatten = tf.keras.layers.Flatten()(preprocessed_text["input_word_ids"])
        dense_layer1 = tf.keras.layers.Dense(1024, activation="relu")(flatten)
        dense_layer1 = tf.keras.layers.Dense(1024, activation="relu")(dense_layer1)
        dropout_layer1 = tf.keras.layers.Dropout(0.1)(dense_layer1)
        output = tf.keras.layers.Dense(3, activation='softmax', name="output")(dropout_layer1)
        self.model = tf.keras.Model(inputs=[text_input], outputs = [output])
        self.model._name = "student_distilled"

        # Student model summary
        self.model.summary()
        print("Student model created ----")
        print("********************************************************")

    
    def get_model_summary(self):
        """
        Prints a summary of the model
        """
        print(self.model.summary())
    
    def compile_model(self):
        """
        Compile the model
        """
        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]

        self.model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=METRICS)
    
    def train_model(self, X_train, y_train, epochs = 10):
        """
        Train the model.
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of epochs
        """
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        self.model.fit(X_train, np.asarray(y_train), epochs = epochs, callbacks = [callback])

    def save_mode(self, data_path = "trained_model/"):
        """
        Save the model
        Args:
            data_path: Path to save the model
        """
        self.model.save(data_path + self.model_name)
    
    def load_model(self, data_path = "trained_model/"):
        """
        Load the model
        Args:
            data_path: Path to load the model
        """
        self.model = tf.keras.models.load_model(data_path + self.model_name)
        try:
            print("Loading model: " + data_path + self.model_name)
            self.model = tf.keras.models.load_model(data_path + self.model_name)
        except:
            print("Model not found")
    
    def save_model_weights(self, data_path = "trained_model/trained_model_weights/"):
        """
        Save the model weights
        Args:
            data_path: Path to save the model weights
        """
        self.model.save_weights(data_path + self.model_name + "_weights.h5")
    
    def load_model_weights(self, data_path = "trained_model/trained_model_weights/"):
        """
        Load the model weights
        Args:
            data_path: Path to load the model weights
        """
        self.model.load_weights(data_path + self.model_name + "_weights.h5")
    
    def predict_model(self, X):
        """
        Predict the model
        Args:
            X: Data to predict
        """
        start = time.time()
        if type(X) != list:
            X = [X]
        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        elapsed = time.time() - start
        return y_pred, elapsed

    def get_number_of_total_parameters(self):
        """
        Get the number of total parameters in the model
        """
        return self.model.count_params()

    def get_number_of_trainable_parameters(self):
        """
        Get the number of trainable parameters in the model
        """
        return np.sum([np.prod(param.get_shape().as_list()) for param in self.model.trainable_variables])

    def get_number_of_non_trainable_parameters(self):
        """
        Get the number of non trainable parameters in the model
        """
        return np.sum([np.prod(param.get_shape().as_list()) for param in self.model.non_trainable_variables])

if "__name__" == "main":
    model = ModelTraining("bert12")