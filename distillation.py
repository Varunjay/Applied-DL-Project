# Referred from Keras.io
# Link: https://keras.io/examples/vision/knowledge_distillation/

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from data_prep import DataPrep

class Distiller(tf.keras.Model):
    def __init__(self, teacher_weight = "bert12", student_name = "Conv", teacher_name = "Bert"):
        super(Distiller, self).__init__()
        self.teacher_name = teacher_name
        self.student_name = student_name
        self.teacher_weight = teacher_weight

        self.maxlen = 200

    def create_teacher(self, layer = "12_H-768_A-12/2", data_path = "trained_model/trained_model_weights/"):
        """
        Create the teacher model and load the weights.
        """
        # Loading Bert model preprocessing
        bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

        # Loading Bert model
        print("Getting model from: " + "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-" + layer)
        bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-" + layer)

        # Creating the teacher model
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)
        layer1 = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
        layer2 = tf.keras.layers.Dense(3, activation='softmax', name="output")(layer1)

        # Use inputs and outputs to construct a final teacher model
        self.teacher = tf.keras.Model(inputs=[text_input], outputs = [layer2])
        self.teacher._name = self.teacher_name

        # Teacher model summary
        self.teacher.summary()

        # Load the weights
        self.teacher.load_weights(data_path + "model" + self.teacher_weight + "_weights.h5")

    def create_student(self, data):
        """
        Create the student model.
        """
        vectorize_layer = tf.keras.layers.TextVectorization(
                                        max_tokens=1024,
                                        output_mode='int',
                                        output_sequence_length=768)
        vectorize_layer.adapt(data)
        # Creating the Student model
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='input')
        vectorized = vectorize_layer(text_input, training=False)
        embedding = tf.keras.layers.Embedding(256, 768)(vectorized)
        Conv1D = tf.keras.layers.Conv1D(32, 8, activation="relu")(embedding)
        MaxPooling1D = tf.keras.layers.MaxPooling1D(2)(Conv1D)
        Conv1D_1 = tf.keras.layers.Conv1D(32, 8, activation="relu")(MaxPooling1D)
        MaxPooling1D_1 = tf.keras.layers.MaxPooling1D(2)(Conv1D_1)
        flatten = tf.keras.layers.Flatten()(MaxPooling1D_1)
        dense_layer1 = tf.keras.layers.Dense(1024)(flatten)
        dropout_layer1 = tf.keras.layers.Dropout(0.1)(dense_layer1)
        output = tf.keras.layers.Dense(3, activation='softmax', name="output")(dropout_layer1)

        # Use inputs and outputs to construct a final student model
        self.student = tf.keras.Model(inputs=[text_input], outputs = [output])
        self.student._name = self.student_name

        # Student model summary
        self.student.summary()

    # def set_text_tokenizer(self, tokenizer):
    #     self.tokenizer = tokenizer

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        """
        Setup the model for training.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        
    def train_step(self, data):

        X_train , y_train = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(X_train, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(X_train, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y_train, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y_train, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
    
    def get_student_model(self):
        return self.student

    def get_teacher_model(self):
        return self.teacher
    
    def save_student_model_weights(self, name = "distilled", path = "trained_model/"):
        self.student.save(path + "model" + name + self.student_name)
    def train_student(self, X_train, y_train, epochs=100):
        self.student.fit(X_train, y_train, epochs=epochs, batch_size=256)