import os
import torch
from torch.utils.data import (TensorDataset, random_split,
                              RandomSampler, DataLoader,
                              SequentialSampler)
from torchtext.legacy import data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from helper import *
from Logger import Logger
import time

def device():
    """
    Set device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(text, bert_prob, real_label):
    """
    Set device for text, bert_prob, real_label

    Args:
        text: list of text
        bert_prob: list of bert_prob
        real_label: list of real_label
    """
    text = text.to(device())
    bert_prob = bert_prob.to(device())
    real_label = real_label.to(device())
    return text, bert_prob, real_label



class Baseline(Logger):
    """
    Baseline model for training
    """

    # Set name of model, weights, vocab
    vocab_name = None
    weights_name = None
    model_name = None

    def model(self, text_field):
        """
        Create the model

        Args:
            text_field: text field
        """
        raise NotImplementedError()

    @staticmethod
    def to_dataset(text, y_pred, y_true):
        """
        Convert the data to TensorDataset

        Args:
            text: list of words
            y_pred: list of probabilities
            y_true: list of labels
        """
        text = torch.LongTensor(text)
        y_pred = torch.LongTensor(y_pred)
        y_true = torch.LongTensor(y_true)
        return torch.utils.data.TensorDataset(text, y_pred, y_true)

    @staticmethod
    def to_device(text, bert_prob, real_label):
        """
        Change device

        Args:
            text: list of words
            bert_prob: list of probabilities
            real_label: list of labels
        """
        text = text.to(device())
        bert_prob = bert_prob.to(device())
        real_label = real_label.to(device())
        return text, bert_prob, real_label
    
    def train(self, texts, y_pred, y_true, output_dir):
        """
        Train the model

        Args:
            texts: list of words
            y_pred: list of probabilities
            y_true: list of labels
            output_dir: output directory
        """
        texts = [text.split() for text in texts]

        # Train/validation split
        X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(texts, y_pred, y_true, 
                                                                                test_size=self.config['test_size'], 
                                                                                stratify=y_true,random_state=42)

        # Build vocab
        text_field = data.Field()
        text_field.build_vocab(X_train, max_size=self.config['max_vocab_size'])

        # Save vocab
        self.save_vocab(text_field, output_dir)

        # Pad sequences
        X_train_pad = [pad_sequence(text, self.config['max_seq_length']) for text in X_train]
        X_test_pad = [pad_sequence(text, self.config['max_seq_length']) for text in X_test]

        # Get indexes of padded sequences
        X_train_index = [to_index(text_field.vocab, text) for text in X_train_pad]
        X_test_index = [to_index(text_field.vocab, text) for text in X_test_pad]

        # Get tensor dataset
        train_dataset = self.to_dataset(X_train_index, y_train, y_true_train)
        test_dataset = self.to_dataset(X_test_index, y_test, y_true_test)

        # Model
        model = self.model(text_field)
        model.to(device())

        # Train the model
        self.trainmodel(model, train_dataset, test_dataset, output_dir)

        # Save the model
        # self.save(model, os.path.join(output_dir, self.model_name))
        # self.save(model.state_dict(), os.path.join(output_dir, self.weights_name))
        text_field.vocab.save_vocab(os.path.join(output_dir, self.vocab_name))

        return model, text_field.vocab
    
    @staticmethod
    def optimizer(model):
        """
        Optimizer

        Args:
            model: model
        """
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return optimizer, scheduler

    def trainmodel(self, model, train_dataset, test_dataset, output_dir):
        """
        Train the model

        Args:
            model: model
            train_dataset: train dataset
            test_dataset: test dataset
            output_dir: output directory
        """
        
        # Set number of epochs and best eval score
        epochs = self.config['epochs']
        best_eval_loss = 100000

        for epoch in range(epochs):

            start = time.time()

            print("Epoch: {}".format(epoch))

            # Train one epoch
            train_loss = self.train_epoch(model, train_dataset)

            # Evaluate on test set
            eval_loss, eval_acc = self.eval_epoch(model, test_dataset, epoch)

            epoch_time = time.time() - start

            print("Train loss: {:.4f}, Eval loss: {:.4f}, Eval acc: {:.4f}, Epoch time: {:.4f}".format(train_loss, eval_loss, eval_acc, epoch_time))
            self.log_epoch(epoch, train_loss, eval_loss, eval_acc, epoch_time)

            # Save the model if the validation loss is the best we've seen so far.
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_model(model, output_dir)
                print("Saving model...")
            else:
                print("No improvement...")
        
    def train_epoch(self, model, train_dataset):
        """
        Train the model for one epoch on the train dataset

        Args:
            model: model
            train_dataset: train dataset
        """
        
        # Create data loader
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.config['batch_size'], drop_last=True)

        model.train()

        print("Number of training batches: {}".format(len(train_dataloader)))
        print("Number of trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        print("Total samples: {}".format(len(train_dataloader.dataset)))
        print("Running on device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))

        # Get optimizer and scheduler
        optimizer, scheduler = self.optimizer(model)

        train_loss = 0.0
        total_samples = len(train_dataloader.dataset)

        # Train the model
        for _, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            text, bert_prob, real_label = to_device(*batch)
            model.zero_grad()

            # Forward pass
            output = model(text.t()).squeeze(1)

            # Compute loss and backprop
            loss = self.loss(output, bert_prob, real_label)
            loss.backward()
            optimizer.step()

            # Update total loss
            train_loss += loss.item()
        
        scheduler.step()
        return train_loss / total_samples
    
    def eval_epoch(self, model, test_dataset, epoch):
        """
        Evaluate the model on the test set

        Args:
            model: model
            test_dataset: test dataset
        """
        # Create data loader
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=self.config['batch_size'], drop_last=True)

        print("Number of test batches: {}".format(len(test_dataloader)))
        print("Total samples to evaluate: {}".format(len(test_dataloader.dataset)))

        # Evaluate the model
        model.eval()

        # Set loss and accuracy
        eval_loss = 0
        accuracy = 0
        total_samples = len(test_dataloader.dataset)
        predictions = None
        labels = None

        # Evaluate the model
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):

                text, bert_prob, real_label = to_device(*batch)

                # Forward pass
                output = model(text.t()).squeeze(1)

                # Compute loss
                loss = self.loss(output, bert_prob, real_label)
                eval_loss += loss.item()

                # Compute accuracy
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                accuracy += torch.sum(preds == real_label).item()

                predictions, labels = self.stack(predictions, preds, real_label, labels)

            self.log_predictions(epoch, predictions, labels)
        return eval_loss / total_samples, accuracy / total_samples

    def predict(self, model, text_field, text):
        """
        Predict the label of the text
        """
        # pad the text
        text_pad = pad_sequence(text, self.config['max_seq_len'])

        # Get indexes of padded sequences
        text_index = to_index(text_field.vocab, text_pad)

        # Get tensor dataset
        dataset = self.to_dataset(text_index)

        # Predict the label
        model.eval()
        with torch.no_grad():

            # Get the output
            output = model(dataset.t()).squeeze(1)
            probs = torch.softmax(output)
            preds = torch.argmax(probs, dim=1)
            pred = preds.item()
        
        return pred

    def loss(self, output, bert_prob, real_label):
        """
        Loss function
        """
        raise NotImplementedError()



            



