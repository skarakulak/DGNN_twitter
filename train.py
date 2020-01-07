import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import pandas as pd
import datetime
import random
from math import ceil, sqrt
from collections import namedtuple, defaultdict
from tqdm import tqdm
from scipy import stats
from sklearn import preprocessing, metrics
import os
import glob
import copy
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int, default=1)
parser.add_argument('num_epochs', type=int, default=10)
parser.add_argument('detach_at', type=int, default=4)
parser.add_argument('in_feats', type=int, default=325)
parser.add_argument('n_hidden_gcn', type=int, default=64)
parser.add_argument('n_hidden_lstm', type=int, default=64)
parser.add_argument('n_hidden_gru', type=int, default=64)
parser.add_argument('n_classes', type=int, default=1)
parser.add_argument('n_layers_pre_rnn', type=int, default=1)
parser.add_argument('n_layers_post_rnn', type=int, default=1)
parser.add_argument('dropout', type=float, default=.1)
parser.add_argument('lr', type=float, default=0.01)
parser.add_argument('weight_decay', type=float, default=0.000001)
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tweet_categs = Categories(['own_tweet', 'retweeted_by', 'replied_by', 'quoted_by', 'retweeted', 'replied', 'quoted'])
label_categs = Categories(['normal', 'hateful'])

user_dict_ds = torch.load('./user_dict_full.pkl')

user_mapping = Mapping([v.uID for k, v in user_dict_ds.items()])
num_users = len(user_mapping)
timestamps = sorted(list(set.union(*[set(v.timesteps.keys()) for k,v in user_dict_ds.items() ] )))
tweet_dim = user_dict_ds[user_mapping.ind2ID(0)].tweetvecs[0].size(0)

# extract user features, labels and the training mask
l_x, l_y, mask_y = [], [], []
for i in tqdm(range(num_users)):
    u = user_dict_ds[user_mapping.ind2ID(i)]
    l_x.append(np.concatenate((u.features[3:12],u.features[206:])))
#     l_x.append(u.features)
    mask_y.append(True if u.label is not None else False)
    l_y.append(u.label if u.label is not None else -1)
    for k in u.tweetvecs:
        k.requires_grad = False
        k = k.to('cpu')


temp_cpoint = torch.load('./edges_over_time.pkl')
df_upd_add_edge = temp_cpoint['df_upd_add_edge']
df_upd_existing_edge = temp_cpoint['df_upd_existing_edge']
del temp_cpoint
edge_mapping = Mapping([
    (uind, linkeduind) for uind, linkeduind
    in df_upd_add_edge[['uInd', 'linked_uInd']].itertuples(index=False)
])


empty_tweet = nn.Parameter(
    torch.rand(tweet_dim, device=device)*2./sqrt(tweet_dim)-1./sqrt(tweet_dim)
).to(device)
for i in tqdm(range(num_users)):
    u = user_dict_ds[user_mapping.ind2ID(i)]
    for k in u.tweetvecs:
        k.requires_grad = False
        k = k.to('cuda')
    if len(u.tweetvecs) > len(u.timesteps):
        u.tweetvecs[-1] = empty_tweet

# GRAPH
class GCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 ):
        super().__init__()
        self.g = g
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout else 0.
        self.linear_self = nn.Linear(in_feats, out_feats)
        self.linear_node = nn.Linear(in_feats, out_feats * 6)
        self.linear_edge = nn.Linear(6, out_feats, bias=False)  # separately for src and dst
        self.node_apply_activation = activation
        stdev = 1. / sqrt(out_feats)
        self.bias = nn.Parameter(torch.randn(1, out_feats) * 2 * stdev - stdev)

    #         self.gcn_apply_gru_cell = nn.GRUCell(out_feats, out_feats)

    def gcn_msg(self, edge):
        msg = edge.data['h_degrees']
        # for each edge type
        for i in range(6):
            msg = msg + \
                  edge.src['h_node_msg'][:, i * self.out_feats:(i + 1) * self.out_feats] * \
                  (edge.data['degrees'][:, i] > 0).float()[:,None]
        msg = msg * edge.src['norm']
        return {'m': msg}

    def gcn_reduce(self, node):
        accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
        return {'h': accum}

    def node_apply(self, nodes):
        h = nodes.data['h'] + nodes.data['h_node_self']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}

    def forward(self, h):
        #         if self.dropout: h = self.dropout(h)
        self.g.ndata['h_node_msg'] = self.linear_node(h)
        self.g.ndata['h_node_self'] = self.linear_self(h)
        self.g.edata['h_degrees'] = self.linear_edge(self.g.edata['degrees'])

        self.g.update_all(self.gcn_msg, self.gcn_reduce, self.node_apply)
        h = self.g.ndata.pop('h')
        return h


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden_gcn,
                 n_hidden_gru,
                 n_hidden_lstm,
                 n_classes,
                 n_layers_pre_rnn,
                 n_layers_post_rnn,
                 activation,
                 dropout):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden_gru = n_hidden_gru
        self.n_hidden_lstm = n_hidden_lstm
        self.n_hidden_gcn = n_hidden_gcn
        self.n_classes = n_classes

        self.layers_pre_rnn = nn.ModuleList()
        self.layers_post_rnn = nn.ModuleList()

        # input layer
        self.layers_pre_rnn.append(GCNLayer(g, in_feats, n_hidden_gcn, activation, dropout))
        # hidden layers pre lstm
        for i in range(n_layers_pre_rnn - 1):
            self.layers_pre_rnn.append(GCNLayer(g, n_hidden_gcn, n_hidden_gcn, activation, dropout))
        # rnn cells
        self.gru_cell_emb = nn.GRUCell(n_hidden_gcn, n_hidden_gru)
        self.lstm_cell = nn.LSTMCell(n_hidden_gcn, n_hidden_lstm, )
        # initial layer post lstm
        self.layers_post_rnn.append(GCNLayer(g, n_hidden_lstm, n_hidden_gcn, activation, dropout))
        # hidden layers post lstm
        for i in range(n_layers_post_rnn - 1):
            self.layers_post_rnn.append(GCNLayer(g, n_hidden_gcn, n_hidden_gcn, activation, dropout))
        # output layer

    #         self.linear_emb = nn.Linear(n_hidden_gcn, in_feats)
    #         self.layers_post_rnn.append(GCNLayer(g, n_hidden_gcn, n_classes, None, dropout))

    def forward(self, features, hidden_gru, hidden_states):  # , existing_nodes):
        #         ipdb.set_trace()
        h = features  # [existing_nodes]
        for layer in self.layers_pre_rnn:
            h = layer(h)
        h_lstm, c_lstm = self.lstm_cell(h, hidden_states)  # [existing_nodes])
        h = self.layers_post_rnn[0](h_lstm)
        for layer in self.layers_post_rnn[1:]:
            h = layer(h)
        #         h_e = self.linear_emb(h)
        hidden_gru = self.gru_cell_emb(h, hidden_gru)  # [existing_nodes])
        return hidden_gru, (h_lstm, c_lstm)

stdev = 1./sqrt(args.n_hidden_lstm)
lstm_h_init = nn.Parameter(torch.rand(args.n_hidden_lstm, device=device)*2*stdev-stdev)
lstm_c_init = nn.Parameter(torch.rand(args.n_hidden_lstm, device=device)*2*stdev-stdev)
stdev = 1./sqrt(args.n_hidden_gru)
gru_h_init = nn.Parameter(torch.rand(args.n_hidden_gru, device=device)*2*stdev-stdev)

# init graph
G = dgl.DGLGraph()
G.add_nodes(num_users)

G.ndata['x'] = torch.Tensor(np.stack(l_x)).to(device)
G.ndata['y'] = torch.Tensor(l_y).to(device)
G.ndata['y_mask'] = torch.BoolTensor(mask_y).to(device)


model = GCN(
    G,
    args.in_feats,
    args.n_hidden_gcn,
    args.n_hidden_gru,
    args.n_hidden_lstm,
    args.n_classes,
    args.n_layers_pre_rnn,
    args.n_layers_post_rnn,
    nn.LeakyReLU(.01),
    args.dropout).to(device)
MLP = nn.Linear(args.n_hidden_gru, args.n_classes).to(device)

temp_mask = torch.distributions.Binomial(1,.8)
training_mask_temp = temp_mask.sample([num_users])
training_mask = (torch.Tensor(mask_y) * training_mask_temp).bool().to(device)
valid_mask = (torch.Tensor(mask_y) * (1-training_mask_temp)).bool().to(device)

criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
optimizer = torch.optim.Adam(
    list(model.parameters()) + [lstm_h_init, lstm_c_init, gru_h_init] +
    list(MLP.parameters()), lr=3e-3, weight_decay=args.weight_decay
)

degree_cols = [str(k) for k in range(1,7)]

for epoch in range(args.num_epochs):
    G.clear()
    G.add_nodes(num_users)
    G.ndata['x'] = torch.Tensor(np.stack(l_x)).to(device)
    G.ndata['y'] = torch.Tensor(l_y).to(device)
    #     G.ndata['y_mask'] = torch.BoolTensor(mask_y).to(device)

    t = timestamps[0]

    df_t = df_upd_add_edge.loc[t]
    G.add_edges(
        df_t.uInd.to_list(),
        df_t.linked_uInd.to_list(),
        data={'degrees': torch.Tensor(df_t[degree_cols].values).to(device)}
    )
    calc_norms(G, device)

    h_gru, (h_lstm, c_lstm) = model(
        G.ndata['x'],
        gru_h_init.expand(num_users, args.n_hidden_gru),
        (lstm_h_init.expand(num_users, args.n_hidden_lstm),
         lstm_c_init.expand(num_users, args.n_hidden_lstm)))

    loss = 0
    edge_num = 0
    for i, t in enumerate(timestamps[1:]):
        df_t = df_upd_add_edge.loc[t]
        G.add_edges(
            df_t.uInd.to_list(),
            df_t.linked_uInd.to_list(),
            data={'degrees': torch.Tensor(df_t[degree_cols].values).to(device)}
        )
        calc_norms(G)

        df_upt_t = df_upd_existing_edge.loc[t]
        loc_idx = torch.LongTensor(df_upt_t.loc[t].edge_id.values)
        G.edata['degrees'][loc_idx] = G.edata['degrees'][loc_idx] + torch.Tensor(df_upt_t[degree_cols].values).to(
            device)

        h_gru, (h_lstm, c_lstm) = model(G.ndata['x'], h_gru, (h_lstm, c_lstm))

        print(t)

        if ((i + 1) % args.detach_at == 0) or (i + 2 == len(timestamps)):
            # predict edge labels at time `t`
            y_logits = MLP(h_gru).flatten()
            loss = criterion(y_logits[training_mask], G.ndata['y'][training_mask])
            loss += loss * G.ndata['y'][training_mask] * 7
            loss = loss.sum()
            print(f'loss: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            valid_loss = criterion(y_logits[valid_mask], G.ndata['y'][valid_mask])
            valid_loss += valid_loss * G.ndata['y'][valid_mask] * 7
            valid_loss = valid_loss.sum()
            print(f'validation loss: {valid_loss.item()}')
            fpr, tpr, thresholds = metrics.roc_curve(
                G.ndata['y'][valid_mask].cpu().numpy(),
                torch.sigmoid(y_logits[valid_mask]).detach().cpu().numpy())
            auc = metrics.auc(fpr, tpr)
            print(f'auc: {auc}')

            h_gru, h_lstm, c_lstm = h_gru.detach(), h_lstm.detach(), c_lstm.detach()
