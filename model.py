# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 23:14:30 2022

@author: culro
"""

import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from numpy import argmax


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Model:
    """
    Returns the model
    """
    def __init__(self, loc):
        """
        Creates the model and laod the weight
        """
        bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
        # Bert layers
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)

        # Neural network layers
        l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
        l = tf.keras.layers.Dense(3, activation='softmax', name="output")(l)


        # Use inputs and outputs to construct a final model
        self.model = tf.keras.Model(inputs=[text_input], outputs = [l])

        # load weights into new model
        self.model.load_weights(loc)
        
    def prediction(self, review):
        """
        Returns the prediction of the model
        """
        return argmax(self.model.predict([review]), axis=1)