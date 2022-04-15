import os
from pickletools import optimize
import torch
from torch.utils.data import (TensorDataset, random_split,
                              RandomSampler, DataLoader,
                              SequentialSampler)
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
from Logger import Logger
import time

def device():
    """
    Set device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_to_input(batch):
    """
    Convert batch to input
    """
    inputs = {
        'input_ids': batch[0],
        'attention_mask': batch[1],
        'token_type_ids': batch[2],
        'labels': batch[3]
    }
    return inputs

def features_to_input(features):
    """
    Convert features to input
    """
    print("Feature to input: {}".format(features))
    # input = {
    #     'input_ids': torch.tensor(feature.input_ids, dtype=torch.long),
    #     'attention_mask': torch.tensor(feature.input_mask, dtype=torch.long),
    #     'token_type_ids': torch.tensor(feature.segment_ids, dtype=torch.long),
    #     'labels': torch.tensor(feature.label_id, dtype=torch.long)
    # }
    inputs = {
        'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
        'attention_mask': torch.tensor([f.input_mask for f in features], dtype=torch.long),
        'token_type_ids': torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        'labels': torch.tensor([f.label_id for f in features], dtype=torch.long)
    }
    return inputs

class BERTBaseline(Logger):
    """
    BERT baseline
    """

    model_name = "bert_pretrained"
    weights_name = "bert_pretrained_weights.pt"
    tokenizer_weights_name = "bert_pretrained_tokenizer_weights.pt"
    tokenizer_name = "bert_pretrained_tokenizer"


    def train(self, texts, tekenizer, output_dir):
        """
        Create the model and train it

        Args:
            texts: list of list of words
            tekenizer: tokenizer
            output_dir: output directory
        """

        print("\nLoading the bert pretrained model...")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        model.to(device())
        self.trainmodel(texts, tekenizer, model, output_dir)
        return model

    def trainmodel(self, dataset, tokenizer, model, output_dir):
        """
        Train the model

        Args:
            texts: list of list of words
            tokenizer: tokenizer
            model: model
            output_dir: output directory
        """

        # Split the dataset
        train_size = int(len(dataset) * self.config['train_split'])
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        model.zero_grad()

        for epoch in range(int(self.config['epochs'])):

            start = time.time()

            # Train one epoch
            train_loss = self.train_epoch(model, train_dataset, epoch)

            # Evaluate on test set
            eval_loss, eval_acc = self.eval_epoch(model, test_dataset, epoch)

            epoch_time = time.time() - start
            
            print("Train loss: {:.4f}, Eval loss: {:.4f}, Eval acc: {:.4f}, Epoch time: {:.4f}".format(train_loss, eval_loss, eval_acc, epoch_time))
            self.log_epoch(epoch, train_loss, eval_loss, eval_acc, epoch_time)

            # Save the model
            model.save_pretrained(os.path.join(output_dir, self.model_name))
            torch.save(model.state_dict(), os.path.join(output_dir, self.weights_name))
            torch.save(tokenizer.vocab, os.path.join(output_dir, self.tokenizer_weights_name))
            tokenizer.save_pretrained(os.path.join(output_dir, self.tokenizer_name))
        
    def train_epoch(self, model, dataset, epoch):

        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=self.config['batch_size'])

        optimizer = torch.optim.AdamW(model.parameters(),
                          lr=self.config['learning_rate'],
                          eps=self.config['adam_epsilon'])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        train_loss = 0.0

        print("\nNumber of training batches: {}".format(len(train_dataloader)))
        print("Number of trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        print("Total samples: {}".format(len(train_dataloader.dataset)))
        print("Running on device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))

        for _, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device()) for t in batch)
            inputs = batch_to_input(batch)
            outputs = model(**inputs)

            loss = outputs[0]
            loss.backward()

            train_loss += loss.item()

            optimizer.step()
            model.zero_grad()
        
        scheduler.step()

        return train_loss / len(train_dataloader)

    def eval_epoch(self, model, dataset, epoch):

        test_sampler = SequentialSampler(dataset)
        test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=self.config['batch_size'])

        print("Number of test batches: {}".format(len(test_dataloader)))
        print("Total samples to evaluate: {}".format(len(test_dataloader.dataset)))

        # Evaluate the model
        model.eval()

        # Set loss and accuracy
        eval_loss = 0
        accuracy = 0
        total_samples = len(test_dataloader.dataset)

        # Progress
        labels = None
        predictions = None

        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_dataloader, desc="Evaluation")):
                batch = tuple(t.to(device()) for t in batch)
                inputs = batch_to_input(batch)
                outputs = model(**inputs)
                loss = outputs[0]

                eval_loss += loss.item()

                logits = outputs[1]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                accuracy += torch.sum(torch.argmax(probs, dim=1) ==  batch[3]).item()

                predictions, labels = self.stack(predictions, preds, batch[3], labels)
        
        self.log_predictions(epoch, predictions, labels)
        return eval_loss / total_samples, accuracy / total_samples

    def predict(self, model, text):
        """
        Predict the labels

        Args:
            model: model
            dataset: list of list of words
            tekenizer: tokenizer
        """
        model.eval()

        with torch.no_grad():
            inputs = features_to_input(text)
            outputs = model(**inputs)
            logits = outputs[1]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        preds = preds.cpu().numpy().tolist()[0]
        return preds

    def load_model(self, output_dir):
        """
        Load the model

        Args:
            output_dir: output directory
        """
        model = BertForSequenceClassification.from_pretrained(os.path.join(output_dir, self.model_name), num_labels=3)
        model.load_state_dict(torch.load(os.path.join(output_dir, self.weights_name)))
        return model
            
    def load_tokenizer(self, output_dir):
        """
        Load the tokenizer

        Args:
            output_dir: output directory
        """
        tokenizer = BertTokenizer.from_pretrained(os.path.join(output_dir, self.tokenizer_name))
        return tokenizer