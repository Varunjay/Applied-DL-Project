import pandas as pd
from helper import *
from LSTMDistilled import LSTMDistilled as DistilledLSTM
from transformers import BertTokenizer
from Bert import BERTBaseline as Bert
from Bert import batch_to_input
from tqdm import tqdm
from torch.utils.data import (DataLoader, SequentialSampler)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Load teacher Model
    teacher = Bert(DISTILL_BILSTM_CONFIG)
    teacher_model = teacher.load_model(MODEL_PATH)
    teacher_tokenizer = teacher.load_tokenizer(MODEL_PATH)
    print("Loaded teacher model and tokenizer")

    # Load the training data
    train_data = load_data("./Data/train.csv", DISTILL_BILSTM_CONFIG['max_seq_length'], teacher_tokenizer)

    # Creating teacher DataLoader
    teacher_sampler = SequentialSampler(train_data)
    teacher_dataloader = DataLoader(train_data, sampler=teacher_sampler, batch_size=DISTILL_BILSTM_CONFIG['batch_size'])

    # Prepare the teacher model
    teacher_model.to(device())
    teacher_model.eval()

    teacher_logits = None

    for batch in tqdm(teacher_dataloader, desc="Teacher"):
        batch = tuple(t.to(device()) for t in batch)

        with torch.no_grad():
            inputs = batch_to_input(batch)
            outputs = teacher_model(**inputs)
            logits = outputs[1]
            logits = logits.cpu().numpy()

            if teacher_logits is None:
                teacher_logits = logits
            else:
                teacher_logits = np.vstack((teacher_logits, logits))

    # Load the student model data
    data = pd.read_csv("./Data/train.csv")
    X_train = data['review'].values
    y_train = teacher_logits

    y_real = data['rating'].values
    label_dic = {label: i for i, label in enumerate(set([rating for rating in y_real]))}
    y_real = data['rating'].apply(lambda x: float(label_dic[x])).values

    # Create the student model
    student = DistilledLSTM(DISTILL_BILSTM_CONFIG)
    student.train(X_train, y_train, y_real, MODEL_PATH)







