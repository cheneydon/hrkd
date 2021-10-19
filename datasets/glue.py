import os
import csv
import logging
import torch
import datasets
import models
from tqdm import tqdm
from torch.utils.data.dataset import TensorDataset


class GlueExample(object):
    def __init__(self, text_a, text_b, label, id=None, task_id=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.id = id
        self.task_id = task_id


def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        data = []
        for line in reader:
            data.append(line)
        return data


def create_glue_examples(task, glue_dir, split):
    file_name = split + '.tsv'
    if task == 'mrpc':
        data_path = os.path.join(glue_dir, 'MRPC', file_name)
        text_a_id, text_b_id, label_id = [3, 4, 0] if split != 'test' else [3, 4, None]
    elif task == 'mnli':
        data_path = os.path.join(glue_dir, 'MNLI', file_name if split == 'train' else split + '_matched.tsv')
        text_a_id, text_b_id, label_id = [8, 9, -1] if split != 'test' else [8, 9, None]
    elif task == 'mnli-mm':
        data_path = os.path.join(glue_dir, 'MNLI', file_name if split == 'train' else split + '_mismatched.tsv')
        text_a_id, text_b_id, label_id = [8, 9, -1] if split != 'test' else [8, 9, None]
    elif task == 'ax':
        data_path = os.path.join(glue_dir, 'MNLI', 'diagnostic.tsv')
        text_a_id, text_b_id, label_id = [1, 2, None]
    elif task == 'cola':
        data_path = os.path.join(glue_dir, 'CoLA', file_name)
        text_a_id, text_b_id, label_id = [3, None, 1] if split != 'test' else [1, None, None]
    elif task == 'sst-2':
        data_path = os.path.join(glue_dir, 'SST-2', file_name)
        text_a_id, text_b_id, label_id = [0, None, 1] if split != 'test' else [1, None, None]
    elif task == 'sts-b':
        data_path = os.path.join(glue_dir, 'STS-B', file_name)
        text_a_id, text_b_id, label_id = [7, 8, -1] if split != 'test' else [7, 8, None]
    elif task == 'qqp':
        data_path = os.path.join(glue_dir, 'QQP', file_name)
        text_a_id, text_b_id, label_id = [3, 4, 5] if split != 'test' else [1, 2, None]
    elif task == 'qnli':
        data_path = os.path.join(glue_dir, 'QNLI', file_name)
        text_a_id, text_b_id, label_id = [1, 2, -1] if split != 'test' else [1, 2, None]
    elif task == 'rte':
        data_path = os.path.join(glue_dir, 'RTE', file_name)
        text_a_id, text_b_id, label_id = [1, 2, -1] if split != 'test' else [1, 2, None]
    elif task == 'wnli':
        data_path = os.path.join(glue_dir, 'WNLI', file_name)
        text_a_id, text_b_id, label_id = [1, 2, -1] if split != 'test' else [1, 2, None]
    else:
        raise KeyError('task \'{}\' is not valid'.format(task))

    labels = datasets.glue_labels[task]
    label_map = {label: i for i, label in enumerate(labels)}
    data = read_tsv(data_path)

    examples = []
    for i, line in enumerate(data):
        if i == 0 and (split == 'test' or (split != 'test' and task != 'cola')):
            continue
        text_a = line[text_a_id]
        text_b = line[text_b_id] if text_b_id is not None else None
        if split == 'test':
            label = None
        else:
            label = line[label_id]
            label = float(label) if task == 'sts-b' else label_map[label]

        id = int(line[0]) if split == 'test' else None
        task_id = datasets.glue_train_tasks_to_ids[task] if task in datasets.glue_train_tasks else None
        examples.append(GlueExample(text_a, text_b, label, id, task_id))
    return examples


def create_glue_dataset(model_name, task, glue_dir, tokenizer, max_seq_len, split, local_rank, cache_dir=''):
    if model_name in models.bert_models:
        model = 'bert'
    elif model_name in models.xlnet_models:
        model = 'xlnet'
    elif model_name in models.roberta_models:
        model = 'roberta'
    else:
        raise KeyError('model name \'{}\' is not valid'.format(model_name))

    cache_file = os.path.join(cache_dir, 'glue', '_'.join([model, task, split, str(max_seq_len)]))
    if tokenizer.lowercase:
        cache_file = os.path.join(cache_dir, 'glue', '_'.join([model, task, split, str(max_seq_len), 'lowercase']))

    if os.path.exists(cache_file):
        if local_rank == 0:
            logging.info('Loading {} cache file from \'{}\''.format(split, cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        if local_rank == 0:
            logging.info('Creating {} examples'.format(split))
        examples = create_glue_examples(task, glue_dir, split)
        texts_a = [example.text_a for example in examples]
        texts_b = [example.text_b for example in examples]

        if local_rank == 0:
            logging.info('Creating {} dataset'.format(split))
        encoded_inputs = [tokenizer.encode(text_a, text_b)
                          for text_a, text_b in tqdm(zip(texts_a, texts_b), total=len(texts_a), disable=local_rank != 0)]

        if cache_dir and local_rank == 0:
            logging.info('Saving {} cache file to \'{}\''.format(split, cache_file))
            cache_dir = os.path.dirname(cache_file)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'examples': examples, 'encoded_inputs': encoded_inputs}, cache_file)

    token_ids = torch.tensor([inp.token_ids for inp in encoded_inputs], dtype=torch.long)
    segment_ids = torch.tensor([inp.segment_ids for inp in encoded_inputs], dtype=torch.long)
    position_ids = torch.tensor([inp.position_ids for inp in encoded_inputs], dtype=torch.long)
    attn_mask = torch.tensor([inp.attn_mask for inp in encoded_inputs], dtype=torch.long)

    if split == 'test':
        ids = torch.tensor([example.id for example in examples], dtype=torch.long)
        dataset = TensorDataset(token_ids, segment_ids, position_ids, attn_mask, ids)
    else:
        labels = torch.tensor([example.label for example in examples], dtype=torch.float if task == 'sts-b' else torch.long)
        dataset = TensorDataset(token_ids, segment_ids, position_ids, attn_mask, labels)
    return examples, encoded_inputs, dataset
