import torch

class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers, 
                    bidirectional=False, dropout=0.5, batch_size=None, device='gpu'):
        """
        Simple LSTM model

        Args:
            input_size: size of the input
            embedding_size: size of the embedding
            hidden_size: size of the hidden layer
            output_size: size of the output
            n_layers: number of layers
            bidirectional: whether to use bidirectional LSTM
            dropout: dropout rate
            batch_size: batch size
            device: device to use
        """
        super(SimpleLSTM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = torch.nn.Embedding(input_size, embedding_size)

        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, num_layers=n_layers, 
                                    bidirectional=bidirectional, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size*2, output_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        Initialize the hidden state
        """
        if self.batch_size is None:
            self.batch_size = 1
        hidden = (torch.autograd.Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_size)).to(self.device),
                torch.autograd.Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_size)).to(self.device))
        hidden[0].to(self.device)
        hidden[0].to(self.device)
        return hidden
    
    def forward(self, text, text_len = None):
        """
        Forward pass
        
        Args:
            text: input text
            text_len: length of the input text
        """
        self.hidden = self.init_hidden()
        text = self.embedding(text)
        text, self.hidden = self.lstm(text, self.hidden)
        hidden, _ = self.hidden
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        output = self.fc(hidden)
        return output

    def predict(self, text, text_len = None):
        """
        Predict the output

        Args:
            text: input text
            text_len: length of the input text
        """
        self.hidden = self.init_hidden()
        text = self.embedding(text)
        text, self.hidden = self.lstm(text, self.hidden)
        hidden, _ = self.hidden
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        output = self.fc(hidden)
        return output.argmax(1)
    
    def predict_prob(self, text, text_len = None):
        """
        Predict the output probability

        Args:
            text: input text
            text_len: length of the input text
        """
        self.hidden = self.init_hidden()
        text = self.embedding(text)
        text, self.hidden = self.lstm(text, self.hidden)
        hidden, _ = self.hidden
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        output = self.fc(hidden)
        return output