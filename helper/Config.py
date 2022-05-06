

MODEL_PATH = './Model/'

BERT_CONFIG = {
    'max_seq_length': 128,
    'batch_size': 32,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'log_dir': "./Log",
    'comment': 'bert',
    'max_vocab_size': 20000,
    'train_split': 0.9,
    'epochs': 10
}

BILSTM_CONFIG = {
    'test_size': 0.1,
    'batch_size': 32,
    'max_seq_length': 128,
    'max_vocab_size': 20000,
    'embedding_size': 16,
    'hidden_size': 8,
    'num_layers': 1,
    'bidirectional': True,
    'dropout': 0.5,
    'batch_size': 32,
    'epochs': 30,
    'log_dir': "./Log",
    'comment': 'bilstm',
}

DISTILL_BILSTM_CONFIG = {
    'test_size': 0.1,
    'batch_size': 64,
    'max_seq_length': 128,
    'max_vocab_size': 20000,
    'embedding_size': 16,
    'hidden_size': 8,
    'num_layers': 1,
    'bidirectional': True,
    'dropout': 0.3,
    'batch_size': 32,
    'epochs': 100,
    'log_dir': "./Log",
    'comment': 'distilled_bilstm',
    'a': 0.1
}
