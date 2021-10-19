import torch.nn as nn
import datasets
import models


class TinyBertTransformerBlock(nn.Module):
    def __init__(self, config):
        super(TinyBertTransformerBlock, self).__init__()

        self.attention = models.BertAttention(config)
        self.ffn = models.BertFeedForwardNetwork(config)

    def forward(self, hidden_states, attn_mask):
        attn_output, attn_score = self.attention(hidden_states, attn_mask)
        output = self.ffn(attn_output)
        return output, attn_score


class TinyBertSingle(nn.Module):
    def __init__(self, config, use_lm=False):
        super(TinyBertSingle, self).__init__()

        self.use_lm = use_lm
        self.embeddings = models.BertEmbedding(config)
        self.encoder = nn.ModuleList([TinyBertTransformerBlock(config) for _ in range(config.num_layers)])

        # self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
        self.fit_denses = nn.ModuleList([nn.Linear(config.hidden_size, config.fit_size) for _ in range(config.num_layers + 1)])
        if self.use_lm:
            self.lm_head = models.BertMaskedLMHead(config, self.embeddings.token_embeddings.weight)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        # all_ffn_outputs.append(self.fit_dense(output))
        all_ffn_outputs.append(self.fit_denses[0](output))

        for i, layer in enumerate(self.encoder):
            output, attn_output = layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
            # all_ffn_outputs.append(self.fit_dense(output))
            all_ffn_outputs.append(self.fit_denses[i + 1](output))

        if self.use_lm:
            output = self.lm_head(output)
        return output, all_attn_outputs, all_ffn_outputs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


class TinyBert(nn.Module):
    def __init__(self, config, task, return_hid=False):
        super(TinyBert, self).__init__()

        self.task = task
        self.return_hid = return_hid
        self.embeddings = models.BertEmbedding(config)
        self.encoder = nn.ModuleList([TinyBertTransformerBlock(config) for _ in range(config.num_layers)])

        # self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
        self.fit_denses = nn.ModuleList([nn.Linear(config.hidden_size, config.fit_size) for _ in range(config.num_layers + 1)])

        self.cls_pooler = models.BertClsPooler(config)
        num_classes = datasets.multi_domain_num_classes[task]
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        # all_ffn_outputs.append(self.fit_dense(output))
        all_ffn_outputs.append(self.fit_denses[0](output))

        for i, layer in enumerate(self.encoder):
            output, attn_output = layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
            # all_ffn_outputs.append(self.fit_dense(output))
            all_ffn_outputs.append(self.fit_denses[i + 1](output))

        output = self.cls_pooler(output[:, 0])
        output = self.classifier(output).squeeze(-1)
        if self.return_hid:
            return output, all_attn_outputs, all_ffn_outputs
        return output

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


class MetaTinyBert(nn.Module):
    def __init__(self, config, task):
        super(MetaTinyBert, self).__init__()

        self.embeddings = models.BertEmbedding(config)
        self.encoder = nn.ModuleList([TinyBertTransformerBlock(config) for _ in range(config.num_layers)])
        self.fit_denses = nn.ModuleList([nn.Linear(config.hidden_size, config.fit_size) for _ in range(config.num_layers + 1)])

        self.cls_pooler = models.BertClsPooler(config)
        self.classifiers = nn.ModuleList([])
        if task == 'mnli':
            all_domains = datasets.all_mnli_domains
        else:  # task == 'amazon_review'
            all_domains = datasets.all_amazon_review_domains
        for _ in all_domains:
            num_classes = datasets.multi_domain_num_classes[task]
            self.classifiers.append(nn.Linear(config.hidden_size, num_classes))

        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask, task_id):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        cur_ffn_output = self.fit_denses[0](output)
        all_ffn_outputs.append(cur_ffn_output)

        for layer_id, layer in enumerate(self.encoder):
            output, attn_output = layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
            cur_ffn_output = self.fit_denses[layer_id + 1](output)
            all_ffn_outputs.append(cur_ffn_output)

        output = self.cls_pooler(output[:, 0])
        output = self.classifiers[task_id](output).squeeze(-1)
        return output, all_attn_outputs, all_ffn_outputs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
