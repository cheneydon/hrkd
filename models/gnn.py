import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, act=F.elu):
        super(GraphAttentionLayer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.act = act

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self._init_weights()

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.act is not None:
            return self.act(h_prime)
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()

        self.dropout = dropout
        self.nheads = nheads
        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)])
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, hidden_states, adj):
        x = F.dropout(hidden_states, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        # return F.log_softmax(x, dim=1)
        return x


class MetaGraph(nn.Module):
    def __init__(self, hidden_size, num_layers, num_domains, hierarchical, dropout=0.1, alpha=0.2, nheads=8):
        super(MetaGraph, self).__init__()

        self.num_domains = num_domains
        self.hierarchical = hierarchical
        self.all_gnn = nn.ModuleList([GAT(hidden_size, hidden_size, 1, dropout, alpha, nheads) for _ in range(num_layers)])
        if self.hierarchical:
            self.w_attn_1 = nn.Parameter(torch.empty(size=(num_layers, hidden_size, hidden_size)))
            self.w_attn_2 = nn.Parameter(torch.empty(size=(num_layers, num_domains, hidden_size, hidden_size)))
            self._init_weights()

    def forward(self, all_feats):  # all_feats: (n_layer, n_domain, hid_sz)
        all_train_ratios, all_hier_attn_ratios = [], []
        for layer_id, feat in enumerate(all_feats):
            if self.hierarchical and layer_id > 0:
                cur_w = self.w_attn_1[layer_id]  # (hid_sz, hid_sz)
                M1 = torch.matmul(feat, torch.matmul(cur_w, feat.T))  # (n_domain, n_domain)
                attn_ratio1 = torch.softmax(M1, dim=-1)  # (n_domain, n_domain)
                ref_feat = torch.matmul(attn_ratio1, feat)  # (n_domain, hid_sz)

                all_hier_attn_feats = []
                cur_hier_attn_ratios = []
                hier_feats = all_feats[:(layer_id + 1)]  # (layer_id + 1, n_domain, hid_sz)
                for domain_id in range(hier_feats.size(1)):
                    cur_w = self.w_attn_2[layer_id][domain_id]  # (hid_sz, hid_sz)
                    cur_hier_feat = hier_feats[:, domain_id, :]  # (layer_id + 1, hid_sz)

                    cur_ref_feat = ref_feat[domain_id]  # (hid_sz)
                    M2 = torch.matmul(cur_hier_feat, torch.matmul(cur_w, cur_ref_feat))  # (layer_id + 1)
                    attn_ratio2 = torch.softmax(M2, dim=-1)  # (layer_id + 1)
                    cur_hier_attn_feat = torch.matmul(attn_ratio2, cur_hier_feat)  # (hid_sz)

                    all_hier_attn_feats.append(cur_hier_attn_feat)
                    cur_hier_attn_ratios.append(attn_ratio2)
                feat = torch.stack(all_hier_attn_feats, dim=0)  # (n_domain, hid_sz)
                cur_hier_attn_ratios = torch.stack(cur_hier_attn_ratios, dim=0)  # (n_domain, layer_id + 1)
                all_hier_attn_ratios.append(cur_hier_attn_ratios)

            adj = torch.ones(self.num_domains, self.num_domains, device=feat.device)
            gnn_output = self.all_gnn[layer_id](feat, adj).squeeze(-1)
            cur_train_ratios = torch.softmax(gnn_output, dim=-1)
            all_train_ratios.append(cur_train_ratios)
        return all_train_ratios, all_hier_attn_ratios

    def _init_weights(self):
        for i, w in enumerate(self.w_attn_1):
            nn.init.xavier_uniform_(self.w_attn_1[i].data, gain=1.414)
        for i, w in enumerate(self.w_attn_2):
            nn.init.xavier_uniform_(self.w_attn_2[i].data, gain=1.414)
