import os
import time
import torch
import logging
import datasets
import models
from random import shuffle
from tqdm import tqdm
from torch.utils.data.dataset import TensorDataset


class DomainExample(object):
    def __init__(self, text_a, text_b, label_id, domain_id):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label_id
        self.domain_id = domain_id


def create_mnli_examples(domain, data_dir, split):
    data_path = os.path.join(data_dir, 'MNLI', 'train.tsv' if split in ['train', 'dev'] else 'dev_matched.tsv')
    genre_id, text_a_id, text_b_id, label_id = [3, 8, 9, -1]

    labels = datasets.glue_labels['mnli']
    label_map = {label: i for i, label in enumerate(labels)}
    data = datasets.read_tsv(data_path)

    examples = []
    for i, line in enumerate(data):
        if i == 0:
            continue
        genre = line[genre_id]
        text_a = line[text_a_id]
        text_b = line[text_b_id] if text_b_id is not None else None
        label = label_map[line[label_id]]
        if genre != domain:
            continue

        domain_id = datasets.mnli_domains_to_ids[domain]
        examples.append(DomainExample(text_a, text_b, label, domain_id))

    return examples


def create_amazon_review_examples(domain, data_dir):
    data_path = os.path.join(data_dir, domain)
    all_labels = datasets.amazon_review_labels
    label_map = {label: i for i, label in enumerate(all_labels)}

    examples = []
    for label in all_labels:
        cur_path = os.path.join(data_path, label + '.review')
        with open(cur_path, 'r') as f:
            all_data = f.readlines()

        cur_review = ''
        start = False
        for i, data in enumerate(all_data):
            data = data.strip()
            if data == '<review_text>':
                start = True
            elif data == '</review_text>':
                text_a = cur_review.strip()
                text_b = None
                label_id = label_map[label]
                domain_id = datasets.amazon_review_domains_to_ids[domain]
                examples.append(DomainExample(text_a, text_b, label_id, domain_id))

                cur_review = ''
                start = False

            if start and data != '' and data != '<review_text>':
                cur_review += data + ' '

    shuffle(examples)
    return examples


def create_multi_domain_mnli_examples(data_dir, split, train_ratio, dev_ratio):
    all_examples = []
    for domain in datasets.all_mnli_domains:
        if split == 'test':
            cur_examples = create_mnli_examples(domain, data_dir, split)
        else:
            cur_examples = create_mnli_examples(domain, data_dir, split)
            num_train = int(train_ratio * len(cur_examples))
            num_dev = int((1 - dev_ratio) * len(cur_examples))
            if split == 'train':
                cur_examples = cur_examples[:num_train]
            else:
                cur_examples = cur_examples[num_dev:]
        all_examples.append(cur_examples)
    return all_examples


def create_multi_domain_amazon_review_examples(data_dir):
    all_examples = []
    for domain in datasets.all_amazon_review_domains:
        cur_examples = create_amazon_review_examples(domain, data_dir)
        all_examples.append(cur_examples)
    return all_examples


def create_single_domain_dataset(model_name, task, domain, data_dir, tokenizer, max_seq_len, split, local_rank,
                                 train_ratio=0.9, dev_ratio=0.1, cache_dir=''):
    if model_name in models.bert_models:
        model = 'bert'
    elif model_name in models.xlnet_models:
        model = 'xlnet'
    elif model_name in models.roberta_models:
        model = 'roberta'
    else:
        raise KeyError('model name \'{}\' is not valid'.format(model_name))

    if task == 'mnli':
        file_name = '_'.join([model, task, domain, split, str(max_seq_len), str(train_ratio), str(dev_ratio)])
    else:  # task == 'amazon_review'
        file_name = '_'.join([model, task, domain, str(max_seq_len)])
    if tokenizer.lowercase:
        file_name += '_lowercase'
    cache_file = os.path.join(cache_dir, 'single_domain', file_name)

    if os.path.exists(cache_file):
        if local_rank == 0:
            logging.info('Loading {} cache file from \'{}\''.format(split, cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        if local_rank == 0:
            logging.info('Creating {} examples'.format(split))
        if task == 'mnli':
            examples = create_mnli_examples(domain, data_dir, split)
            num_train = int(train_ratio * len(examples))
            num_dev = int((1 - dev_ratio) * len(examples))
            if split == 'train':
                examples = examples[:num_train]
            elif split == 'dev':
                examples = examples[num_dev:]
        else:  # task == 'amazon_review'
            examples = create_amazon_review_examples(domain, data_dir)

        if local_rank == 0:
            logging.info('Creating {} dataset'.format(split))
        texts_a = [example.text_a for example in examples]
        texts_b = [example.text_b for example in examples]
        encoded_inputs = [tokenizer.encode(text_a, text_b)
                          for text_a, text_b in tqdm(zip(texts_a, texts_b), total=len(texts_a), disable=local_rank != 0)]

        if cache_dir and local_rank == 0:
            logging.info('Saving {} cache file to \'{}\''.format(split, cache_file))
            cache_dir = os.path.dirname(cache_file)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'examples': examples, 'encoded_inputs': encoded_inputs}, cache_file)

        if task == 'amazon_review':
            exit(0)  # Avoid different data on different ranks, need to run again to load cache

    if task == 'amazon_review':
        num_train = datasets.amazon_review_split[domain][0]
        num_dev = datasets.amazon_review_split[domain][1]
        if split == 'train':
            examples = examples[:num_train]
            encoded_inputs = encoded_inputs[:num_train]
        elif split == 'dev':
            examples = examples[num_train:(num_train + num_dev)]
            encoded_inputs = encoded_inputs[num_train:(num_train + num_dev)]
        else:  # split == 'test'
            examples = examples[(num_train + num_dev):]
            encoded_inputs = encoded_inputs[(num_train + num_dev):]

    token_ids = torch.tensor([inp.token_ids for inp in encoded_inputs], dtype=torch.long)
    segment_ids = torch.tensor([inp.segment_ids for inp in encoded_inputs], dtype=torch.long)
    position_ids = torch.tensor([inp.position_ids for inp in encoded_inputs], dtype=torch.long)
    attn_mask = torch.tensor([inp.attn_mask for inp in encoded_inputs], dtype=torch.long)
    labels = torch.tensor([example.label for example in examples], dtype=torch.long)
    dataset = TensorDataset(token_ids, segment_ids, position_ids, attn_mask, labels)
    return examples, encoded_inputs, dataset


def create_multi_domain_dataset(model_name, task, data_dir, tokenizer, max_seq_len, split, local_rank,
                                train_ratio=0.9, dev_ratio=0.1, cache_dir=''):
    if model_name in models.bert_models:
        model = 'bert'
    elif model_name in models.xlnet_models:
        model = 'xlnet'
    elif model_name in models.roberta_models:
        model = 'roberta'
    else:
        raise KeyError('model name \'{}\' is not valid'.format(model_name))

    if task == 'mnli':
        file_name = '_'.join([model, task, split, str(max_seq_len), str(train_ratio), str(dev_ratio)])
    else:  # task == 'amazon_review'
        file_name = '_'.join([model, task, str(max_seq_len)])
    if tokenizer.lowercase:
        file_name += '_lowercase'
    cache_file = os.path.join(cache_dir, 'multi_domain', file_name)

    if os.path.exists(cache_file):
        if local_rank == 0:
            logging.info('Loading {} cache file from \'{}\''.format(split, cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        if local_rank == 0:
            logging.info('Creating {} examples'.format(split))
        if task == 'mnli':
            examples = create_multi_domain_mnli_examples(data_dir, split, train_ratio, dev_ratio)
        else:  # task == 'amazon_review'
            examples = create_multi_domain_amazon_review_examples(data_dir)

        if local_rank == 0:
            logging.info('Creating {} dataset'.format(split))
        encoded_inputs = []
        for cur_examples in tqdm(examples, disable=local_rank != 0):
            texts_a = [example.text_a for example in cur_examples]
            texts_b = [example.text_b for example in cur_examples]

            cur_encoded_inputs = [tokenizer.encode(text_a, text_b) for text_a, text_b in
                                  tqdm(zip(texts_a, texts_b), total=len(cur_examples), disable=local_rank != 0)]
            encoded_inputs.append(cur_encoded_inputs)

        if cache_dir and local_rank == 0:
            logging.info('Saving {} cache file to \'{}\''.format(split, cache_file))
            cache_dir = os.path.dirname(cache_file)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'examples': examples, 'encoded_inputs': encoded_inputs}, cache_file)

        if task == 'amazon_review':
            exit(0)  # Avoid different data on different ranks, need to run again to load cache

    if task == 'amazon_review':
        for i, domain in enumerate(datasets.all_amazon_review_domains):
            num_train = datasets.amazon_review_split[domain][0]
            num_dev = datasets.amazon_review_split[domain][1]
            if split == 'train':
                examples[i] = examples[i][:num_train]
                encoded_inputs[i] = encoded_inputs[i][:num_train]
            elif split == 'dev':
                examples[i] = examples[i][num_train:(num_train + num_dev)]
                encoded_inputs[i] = encoded_inputs[i][num_train:(num_train + num_dev)]
            else:  # split == 'test'
                examples[i] = examples[i][(num_train + num_dev):]
                encoded_inputs[i] = encoded_inputs[i][(num_train + num_dev):]

    token_ids = [torch.tensor([inp.token_ids for inp in cur_encoded_inputs], dtype=torch.long) for cur_encoded_inputs in encoded_inputs]
    segment_ids = [torch.tensor([inp.segment_ids for inp in cur_encoded_inputs], dtype=torch.long) for cur_encoded_inputs in encoded_inputs]
    position_ids = [torch.tensor([inp.position_ids for inp in cur_encoded_inputs], dtype=torch.long) for cur_encoded_inputs in encoded_inputs]
    attn_mask = [torch.tensor([inp.attn_mask for inp in cur_encoded_inputs], dtype=torch.long) for cur_encoded_inputs in encoded_inputs]
    domain_ids = [torch.tensor([example.domain_id for example in cur_examples], dtype=torch.long) for cur_examples in examples]
    labels = [torch.tensor([example.label for example in cur_examples], dtype=torch.long) for cur_examples in examples]

    all_datasets = [
        TensorDataset(cur_domain_ids, cur_token_ids, cur_segment_ids, cur_position_ids, cur_attn_mask, cur_labels)
        for cur_domain_ids, cur_token_ids, cur_segment_ids, cur_position_ids, cur_attn_mask, cur_labels in
        zip(domain_ids, token_ids, segment_ids, position_ids, attn_mask, labels)]
    return examples, encoded_inputs, all_datasets
