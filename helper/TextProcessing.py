import torch
#from torchtext import data

def get_vocab(texts):
    """
    Create the vocabulary

    Args:
        texts: list of words
    """
    text = [texts.split() for text in texts]
    text_field = torch.utils.data.Field(text, max_size = 10000)
    return text_field

def pad_sequence(text, max_len):
    """
    Pad sequences

    Args:
        text: list of words
        max_len: maximum length
    """
    if len(text) > max_len:
        text = text[:max_len]
    else:
        text = text + ["<pad>"] * (max_len - len(text))
    return text

def to_index(vocab, words):
    """
    Get the index of words

    Args:
        vocab: vocabulary
        words: list of words
    """
    return [vocab.stoi[word] for word in words]

def to_index_predict(vocab, words):
    """
    Get the index of words

    Args:
        vocab: vocabulary
        words: list of words
    """
    return [vocab[word] for word in words if word in vocab]

def to_dataset(text, y_pred, y_true):
    """"
    Convert the data to TensorDataset

    Args:
        text: list of words
        y_pred: list of probabilities
        y_true: list of labels
    """
    text = torch.tensor(text, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.float)
    y_true = torch.tensor(y_true, dtype=torch.long)
    return torch.utils.data.TensorDataset(text, y_pred, y_true)
