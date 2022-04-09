import pandas as pd
import numpy as np
import os

class DataPrep:
    def __init__(self, data_path = None):

        self.data_path = os.getcwd() + data_path
        print("Current path: ", self.data_path)

    def load_data(self, train = True):
        
        self.train = True

        if self.train:
            print("Loading training data")
            try:
                self.X_train = pd.read_csv(self.data_path + 'X_train.csv').squeeze()
            except:
                print("Unable to load X_train.csv")
            
            try:
                self.y_train = pd.read_csv(self.data_path + 'y_train.csv').squeeze()
            except:
                print("Unable to load y_train.csv")
            
        else:
            print("Loading testing data")
            try:
                self.X_test = pd.read_csv(self.data_path + 'X_test.csv').squeeze()
            except:
                print("Unable to load X_test.csv")

            try:
                self.y_test = pd.read_csv(self.data_path + 'y_test.csv').squeeze()
            except:
                print("Unable to load X_test.csv")

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

    def get_sample_data(self, sample_size = 0.01):
        if self.train:
            return self.X_train.sample(frac = sample_size), self.y_train.sample(frac = sample_size)

        else:
            return self.X_test.sample(frac = sample_size), self.y_test.sample(frac = sample_size)
    
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
        if self.train:
            try:
                self.dummy_y_train = np.asarray(pd.get_dummies(self.y_train))
            except:
                print("Unable to get dummy data for y_train")
        else:
            try:
                self.dummy_y_test = np.asarray(pd.get_dummies(self.y_test))
            except:
                print("Unable to get dummy data for y_test")

    def del_data(self):
        del self.X_train
        del self.X_test
        del self.y_train
        del self.y_test
        del self.dummy_y_train
        del self.dummy_y_test
    
if __name__ == '__main__':
    data_prep = DataPrep("/Data/")
    data_prep.load_data()
    data_prep.get_dummy_data()
    print(data_prep.get_data_shape('X_train'))
    print(data_prep.get_data_shape('y_train'))
    print(data_prep.get_sample_data(0.0001))

        

    


        


    

        