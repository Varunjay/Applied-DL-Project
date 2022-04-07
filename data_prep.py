import pandas as pd
import numpy as np
from numpy import argmax
from yaml import load

class DataPrep:
    def __init__(self, data_path = None):

        self.data_path = data_path

    def load_data(self):
        try:
           self.X_train = pd.read_csv(self.data_path + 'X_train.csv').squeeze()
        except:
            print("Unable to load X_train.csv")
        
        try:
            self.X_test = pd.read_csv(self.data_path + 'X_test.csv').squeeze()
        except:
            print("Unable to load X_train.csv")

        try:
            self.y_train = pd.read_csv(self.data_path + 'y_train.csv').squeeze()
        except:
            print("Unable to load X_train.csv")

        try:
            self.y_test = pd.read_csv(self.data_path + 'y_test.csv').squeeze()
        except:
            print("Unable to load X_train.csv")

    def get_data(self, data):
        if data == 'X_train':
            return self.X_train
        elif data == 'X_test':
            return self.X_test
        elif data == 'y_train':
            return self.y_train
        elif data == 'y_test':
            return self.y_test
        else:
            print("Invalid data type")
    
    def get_data_shape(self, data):
        if data == 'X_train':
            return self.X_train.shape
        elif data == 'X_test':
            return self.X_test.shape
        elif data == 'y_train':
            return self.y_train.shape
        elif data == 'y_test':
            return self.y_test.shape
        else:
            print("Invalid data type")

    def get_dummy_data(self):
        try:
            self.dummy_y_train = np.asarray(pd.get_dummies(self.y_train))
        except:
            print("Unable to get dummy data for y_train")
        try:
            self.dummy_y_test = np.asarray(pd.get_dummies(self.y_test))
        except:
            print("Unable to get dummy data for y_test")
    
if __name__ == '__main__':
    data_prep = DataPrep("/Data/")
    data_prep.load_data()
    data_prep.get_dummy_data()
    print(data_prep.get_data_shape('X_train'))
    print(data_prep.get_data_shape('X_test'))
    print(data_prep.get_data_shape('y_train'))
    print(data_prep.get_data_shape('y_test'))

        

    


        


    

        