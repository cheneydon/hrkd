
class BertBaseConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 12
        self.pad_token_id = 0
        self.sep_token_id = 102


class BertLargeConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 1024
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 16
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 4096
        self.num_layers = 24
        self.pad_token_id = 0
        self.sep_token_id = 102


class TinyBertConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 312
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 1200
        self.num_layers = 4
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.fit_size = 768
