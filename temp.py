import logging

import yaml
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter
from model import DocumentRNNEmbeddingsWithAttention
import torch
import flair

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
flair.device = device


def read_csv_classification_corpus(data_folder, train_file=None, dev_file=None, test_file=None, column_name_map=None, delimiter='\t', skip_header=False, label_type='classification'):
    return CSVClassificationCorpus(
        data_folder=data_folder,
        train_file=train_file,
        dev_file=dev_file,
        test_file=test_file,
        column_name_map=column_name_map,
        delimiter=delimiter,
        skip_header=skip_header,
        label_type=label_type
    )

with open("iron.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

corpus_config = config['text_classification']['CSVClassificationCorpus']
column_name_map = corpus_config['column_name_map']
data_folder = corpus_config['data_folder']
label_type = corpus_config.get('label_type', 'classification')
corpus = read_csv_classification_corpus(data_folder,
                                        train_file=corpus_config['train_file'],
                                        dev_file=corpus_config['dev_file'],
                                        test_file=corpus_config['test_file'],
                                        column_name_map=column_name_map)

label_dict = corpus.make_label_dictionary(label_type=label_type)

embedding_config = config['embeddings']['TransformerWordEmbeddings-0']
embeddings = TransformerWordEmbeddings(embedding_config['transformer_model'], fine_tune=embedding_config['fine_tune'])

document_embeddings = DocumentRNNEmbeddingsWithAttention(
    embeddings=[embeddings],
    hidden_size=768,
    rnn_layers=8,
    rnn_type="LSTM",
    bidirectional=True,
    dropout=0.5,
)

train_labels = [label.value for sentence in corpus.train for label in sentence.labels]
label_distribution = Counter(train_labels)
labels = np.array(list(label_distribution.keys()))
class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = torch.FloatTensor(class_weights).to("cuda" if torch.cuda.is_available() else "cpu")
label_weights_dict = {label: weight for label, weight in zip(np.unique(train_labels), class_weights)}
embeddings.to(device)

model = TextClassifier(
    document_embeddings,
    label_dictionary=label_dict,  # 添加label_dictionary参数
    label_type=label_type,
    multi_label=False,
).to(device)

trainer = ModelTrainer(model, corpus)
print(f"Train size: {len(corpus.train)}")
print(f"Dev size: {len(corpus.dev)}")
print(f"Test size: {len(corpus.test)}")

train_config = config['train']
optimizer_config = config['optimizer']
optimizer = AdamW(model.parameters(), lr=float(train_config['learning_rate']), weight_decay=float(optimizer_config['weight_decay']))
train_batch_size = train_config['mini_batch_size']
train_data_size = len(corpus.train) + (len(corpus.dev) if train_config['train_with_dev'] else 0)
train_num_batches = train_data_size // train_batch_size + (train_data_size % train_batch_size > 0)

total_training_steps = len(corpus.train) // train_config['mini_batch_size'] * train_config['max_epochs']
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=train_config['warmup_steps'],
    num_training_steps=total_training_steps
)

lr_scheduler_config = config['train']['lr_scheduler']
lr_scheduler_type = lr_scheduler_config['type']

if lr_scheduler_type == 'OneCycleLR':
    lr_scheduler = OneCycleLR(optimizer,
    max_lr=lr_scheduler_config['max_lr'],
    epochs=lr_scheduler_config['epochs'],
    steps_per_epoch=train_num_batches)

logger = logging.getLogger("flair.training_utils")
logger.info(f" - custom learning_rate: {train_config['learning_rate']}")

trainer.train(
    base_path=config['target_dir'],
    learning_rate=train_config['learning_rate'],
    mini_batch_size=train_config['mini_batch_size'],
    max_epochs=train_config['max_epochs'],
    patience=train_config['patience'],
    train_with_dev=train_config['train_with_dev'],
    embeddings_storage_mode=train_config['embeddings_storage_mode'],
    optimizer=optimizer,
    scheduler=lr_scheduler
)