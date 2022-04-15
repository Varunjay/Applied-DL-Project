import os
import torch
from Baseline import Baseline
from SimpleLSTM import SimpleLSTM as LSTM

def device():
    """
    Set device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMBaseline(Baseline):
    """
    LSTM baseline
    """
    def __init__(self, config):
        super(LSTMBaseline, self).__init__(config)
        self.model_name = 'lstm'
        self.weights_name = 'lstm_weights.pt'
        self.vocab_name = 'lstm_vocab.pt'

    def model(self, text_field):
        """
        Build the model

        Args:
            text_field: text field
        """
        model = LSTM(input_size=len(text_field.vocab),
                     embedding_size=self.config['embedding_size'],
                     hidden_size=self.config['hidden_size'],
                     output_size=3,
                     n_layers=self.config['num_layers'],
                     bidirectional=self.config['bidirectional'],
                     dropout=self.config['dropout'],
                     batch_size=self.config['batch_size'],
                     device=device())

        model.to(device())
        return model

    def loss(self, output, bert_prob, real_label):
        """
        Loss function 

        Args:
            output: output
            bert_prob: bert probability
            real_label: real label
        """
        
        loss = torch.nn.CrossEntropyLoss()
        return loss(output, real_label)

    def save_model(self, model, path):
        """
        Save the model

        Args:
            model: model
            path: path
        """
        torch.save(model.state_dict(), os.path.join(path, self.weights_name))

    def save_vocab(self, text_field, path):
        """
        Save the vocabulary

        Args:
            text_field: text field
            path: path
        """
        torch.save(text_field.vocab, os.path.join(path, self.vocab_name))