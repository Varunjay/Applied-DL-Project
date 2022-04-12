import tensorflow as tf
import numpy as np
from distillation import Distiller
from data_prep import DataPrep

class TrainDistiller:
    def __init__(self, data_loc = "/Data/"):

        self.model = Distiller()

        self.data_prep = DataPrep(data_loc)

    def model_preparation(self):
        self.data_prep.load_data()
        self.data_prep.get_dummy_data()

        self.model.create_student(self.data_prep.X_train)
        self.model.create_teacher()

    def compile_model(self):
        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        self.model.compile(optimizer = tf.keras.optimizers.Adam(), 
                        metrics = METRICS, 
                        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
                        distillation_loss_fn=tf.keras.losses.MeanSquaredError(), 
                        alpha=0.5, 
                        temperature=1)
    

    def fit(self, epochs = 100):
        X_train = self.data_prep.X_train
        y_train = np.asarray(self.data_prep.dummy_y_train)
        self.model.fit(X_train, y_train, epochs)

if __name__ == "__main__":

    print("Training student model")
    model = Distiller(student_name = "baseline")
    data_prep = DataPrep("/Data/")
    data_prep.load_data()
    data_prep.get_dummy_data()
    model.create_student(data_prep.X_train)
    METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    model.student.compile(optimizer = tf.keras.optimizers.Adam(), metrics = METRICS, 
                        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False))
    model.train_student(data_prep.X_train, data_prep.dummy_y_train, epochs = 10)
    model.save_student_model_weights(name = "baseline")


    train_distiller = TrainDistiller()
    train_distiller.model_preparation()
    train_distiller.compile_model()
    train_distiller.fit()
    train_distiller.model.save_student_model_weights()

