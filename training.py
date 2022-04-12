from model_training import ModelTraining
from data_prep import DataPrep

class Train:
    def __init__(self, model_name = "bert12", data_loc = "/Data/", layer = "12_H-768_A-12/2"):

        self.model_name = model_name

        self.model = ModelTraining(self.model_name, layer)
        self.model.create_model()

        print("Model Summary:")
        self.model.get_model_summary()

        # print("Total parameters: ", self.model.get_number_of_total_parameters())
        # print("Total trainable parameters: ", self.model.get_number_of_trainable_parameters())
        # print("Total non-trainable parameters: ", int(self.model.get_number_of_non_trainable_parameters()))

        self.data_prep = DataPrep(data_loc)

    def train(self, X_train, y_train, epochs = 100):
        self.model.compile_model()
        self.model.train_model(X_train, y_train, epochs)

    def save_model(self):
        self.model.save_mode()
    
    def load_training_data(self):
        self.data_prep.load_data()
        self.data_prep.get_dummy_data()

    def start_training(self):
        self.load_training_data()
        self.train(self.data_prep.X_train, self.data_prep.dummy_y_train)

if __name__ == "__main__":
    print("Training now...")
    model = Train("bert12Conv", "/Data/", "12_H-768_A-12/2")
    model.start_training()

    print("Saving model now...")
    model.save_model()

    print("Saving model weights now...")
    model.model.save_model_weights()





    

