import torch
from helper import *
from LSTMBaseline import LSTMBaseline

class LSTMDistilled(LSTMBaseline):
    """
    LSTM distilled
    """
    def __init__(self, config):
        super(LSTMDistilled, self).__init__(config)
        self.model_name = 'lstm_distilled.pt'
        self.weights_name = 'lstm_distilled_weights.pt'
        self.vocab_name = 'lstm_distilled_vocab.pt'
        self.a = self.config['a']
        self.teacher_loss = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()

    def loss(self, output, bert_prob, real_label):
        """
        Loss function 

        Args:
            output: output
            bert_prob: bert probability
            real_label: real labels
        """

        loss = self.a * self.criterion_ce(output, real_label) + (1 - self.a) * self.teacher_loss(output, bert_prob)
        return loss