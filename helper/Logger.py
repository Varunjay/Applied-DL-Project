from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (accuracy_score)



import numpy as np

class Logger(object):

    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(log_dir=config['log_dir'], comment=config['comment'])

    def log_epoch(self, epoch, train_loss, eval_loss, acc, epoch_time):
        self.writer.add_scalar('train/loss', train_loss, epoch)
        self.writer.add_scalar('eval/loss', eval_loss, epoch)
        self.writer.add_scalar('eval/acc', acc, epoch) 
        self.writer.add_scalar('epoch/time', epoch_time, epoch)
    
    def log_predictions(self, epoch, predictions, labels):
        self.writer.add_scalar('eval/acc', np.mean(accuracy_score(labels, predictions)), epoch)

    
    @staticmethod
    def stack(predictions, step_predictions, step_label, labels):
        step_predictions = step_predictions.cpu().numpy()
        step_label = step_label.cpu().numpy()
        if predictions is None:
            predictions = step_predictions
            labels = step_label
        else:
            predictions = np.concatenate((predictions, step_predictions))
            labels = np.concatenate((labels, step_label))
        return predictions, labels
    