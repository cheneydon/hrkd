import models
from .base_tokenizer import BaseTokenizer, EncodedInput
from .bert_tokenizer import BertBasicTokenizer, BertTokenizer


def select_basic_tokenizer(model_name, lowercase, vocab_path):
    if model_name in models.bert_models:
        return BertBasicTokenizer(lowercase, vocab_path)
    else:
        raise KeyError('Basic tokenizer of \'{}\' is not found'.format(model_name))


def select_tokenizer(model_name, lowercase, task, vocab_path, max_seq_len, max_query_len=None):
    if model_name in models.bert_models:
        return BertTokenizer(lowercase, task, vocab_path, max_seq_len, max_query_len)
    else:
        raise KeyError('Tokenizer of \'{}\' is not found'.format(model_name))
