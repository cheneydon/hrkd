from .bert import gelu, BertEmbedding, BertFeedForwardNetwork, BertAttention, \
    BertTransformerBlock, BertClsPooler, BertMaskedLMHead, BertSingle, Bert, MetaBert
from .tiny_bert import TinyBertSingle, TinyBert, MetaTinyBert
from .gnn import MetaGraph
from .configs import BertBaseConfig, BertLargeConfig, TinyBertConfig

bert_models = ['bert_base', 'bert_large', 'meta_bert', 'tiny_bert', 'meta_tinybert']


def select_config(model_name, lowercase=None):
    if model_name in ['bert_base', 'meta_bert']:
        return BertBaseConfig(lowercase)
    elif model_name == 'bert_large':
        return BertLargeConfig(lowercase)
    elif model_name in ['tiny_bert', 'meta_tinybert']:
        return TinyBertConfig(lowercase)
    else:
        raise KeyError('Config for model \'{}\' is not found'.format(model_name))


def select_model(model_name, lowercase, task=None, return_hid=False):
    config = select_config(model_name, lowercase)
    if model_name in bert_models:
        if model_name == 'bert_base':
            return Bert(config, task, return_hid)
        if model_name == 'meta_bert':
            return MetaBert(config, task, return_hid)
        elif model_name == 'tiny_bert':
            return TinyBert(config, task, return_hid)
        elif model_name == 'meta_tinybert':
            return MetaTinyBert(config, task)
        else:
            raise KeyError('Model \'{}\' is not found'.format(model_name))
