from contextlib import closing
import pandas as pd
import torch
from tqdm import tqdm
import pickle

class InputExample(object):
    """
    A single training/test example for simple sequence classification
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Constructs a InputExample

        Args:
            guid: unique id
            text_a: untokenized text of the first sequence
            text_b: untokenized text of the second sequence
            label: label of the example
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """
    A single set of features of data
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        """
        Constructs a InputFeatures

        Args:
            input_ids: list of word ids
            input_mask: list of mask
            segment_ids: list of segment ids
            label_id: label id
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
    
def example_to_feature(example, label_dic, max_seq_length, tokenizer):
    """
    Convert InputExample to InputFeatures

    Args:
        example: list of InputExample
        label_list: list of labels
        max_seq_length: max length of sequence
        tokenizer: tokenizer
    """

    # Tokenize text a
    tokens_a = tokenizer.tokenize(example.text_a)

    # Cut off text a if it is too long
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    # Add [CLS] and [SEP] to tokens_a
    tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]

    # Segment ids for tokens, [CLS] and [SEP]
    segment_ids = [0] * len(tokens)

    # Convert tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Mask for padding
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    # Chech if all length is same and equal to max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # Get label id
    label_id = label_dic[example.label]

    # Return InputFeatures
    return InputFeatures(input_ids, input_mask, segment_ids, label_id)

def feature_to_dataset(features):
    """
    Generate TensorDataset from features

    Args:
        features: list of InputFeatures
    """

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    return torch.utils.data.TensorDataset(input_ids, input_mask, segment_ids, label_ids)

def load_data(data_path, max_seq_length, tokenizer):
    """
    Load data from csv file and convert to InputExample

    Args:
        data_path: path to csv file
        max_seq_length: max length of sequence
        tokenizer: tokenizer
    """

    # Load data from csv file
    df = pd.read_csv(data_path, encoding="utf-8")

    max_percentage = 1
    max_rows = int(len(df) * max_percentage)

    # Convert to InputExample
    examples = []
    for index, row in df.iterrows():
        guid = index
        text_a = row["review"]
        label = row["rating"]
        examples.append(InputExample(guid, text_a, None, label=label))
        if index >= max_rows:
            break

    # Create label dictionary
    label_dic = {label: i for i, label in enumerate(set([example.label for example in examples]))}

    save_pickle(label_dic, "./Model/label_dic.pkl")
 
    # Convert to features
    features = []

    print("Converting to features...")
    for example in tqdm(examples):
        features.append(example_to_feature(example, label_dic, max_seq_length, tokenizer))

    # Generate TensorDataset

    print("Generating TensorDataset...")
    dataset = feature_to_dataset(features)

    return dataset

def save_pickle(data, path = "./Model/"):
    """
    Save data to pickle file

    Args:
        data: data to save
        path: path to save
    """

    with closing(open(path, "wb")) as f:
        pickle.dump(data, f)

def load_pickle(path = "./Model/"):
    """
    Load data from pickle file

    Args:
        path: path to load
    """

    with closing(open(path, "rb")) as f:
        data = pickle.load(f)

    return data

def process_single_text(text, max_seq_length, tokenizer):
    """
    Process single text

    Args:
        text: text to process
    """

    # Tokenize text
    tokens = tokenizer.tokenize(text)

    # Cut off text if it is too long
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
    
    # Add [CLS] and [SEP] to tokens
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

    # Segment ids for tokens, [CLS] and [SEP]
    segment_ids = [0] * len(tokens)

    # Convert tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Mask for padding
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    # Check if all length is same and equal to max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # Return InputFeatures
    return [InputFeatures(input_ids, input_mask, segment_ids, [0])]



    