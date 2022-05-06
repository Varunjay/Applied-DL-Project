from model_training import ModelTraining
from data_prep import DataPrep

class Train:
    def __init__(self, model_name = "bert12", data_loc = "/Data/", layer = "12_H-768_A-12/2"):
        """
        Initialize the training class for off the shelf bert models
        Args:
            model_name: Name of the model to be trained
            data_loc: Location of the data
            layer: Type of bert model to be trained
        """

        self.model_name = model_name

        self.model = ModelTraining(self.model_name, layer)
        self.model.create_model()

        print("Model Summary:")
        self.model.get_model_summary()

        self.data_prep = DataPrep(data_loc)

    def train(self, X_train, y_train, epochs = 100):
        """
        Train the model
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of epochs to train
        """
        self.model.compile_model()
        self.model.train_model(X_train, y_train, epochs)

    def save_model(self):
        """
        Save the model
        """
        self.model.save_mode()
    
    def load_training_data(self):
        """
        Load the training data
        """
        self.data_prep.load_data()
        self.data_prep.get_dummy_data()

    def start_training(self):
        """
        Start the training process
        """
        self.load_training_data()
        self.train(self.data_prep.X_train, self.data_prep.dummy_y_train)

if __name__ == "__main__":
    print("Training now...")
    model = Train("bert12FinedTuned", "/Data/", "12_H-768_A-12/2")
    model.start_training()

    print("Saving model now...")
    model.save_model()

    print("Saving model weights now...")
    model.model.save_model_weights()





    

