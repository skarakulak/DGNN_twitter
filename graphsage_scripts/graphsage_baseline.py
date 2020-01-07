#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import BitcoinOTC
import datetime
from dgl.nn.pytorch import GraphConv
import time
from sklearn.metrics import f1_score
import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import networkx as nx
from dgl import DGLGraph
from dgl.nn.pytorch.conv import SAGEConv
import itertools
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight


# ## Hyperparameters

# In[2]:


num_nodes = 100386


# In[59]:


aggregator_type = 'mean' #mean/gcn/pool/lstm
hid_dims = [128, 256, 512]
defhiddim = 256
n_layers = [1, 2, 3]
defnlayer = 2
dropouts = [0, 0.1, 0.2]
defdropout = 0.1
learning_rate = 0.0003
wt_decays = [3e-7, 3e-6, 3e-5]
defwtdecay = 5e-4
stpsize = 60
checkpt_iter = 5
n_epochs = 100
out_path = '/misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/models/twitter/hyper/'
data_path = '../twitter_data/public/'
activation = F.leaky_relu


# In[46]:


num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    device = 'cuda'
else:
    device = 'cpu'


# ## Data loading

# In[5]:


def load_hate(features, edges, num_features):
    num_nodes = 100386
    num_feats = num_features
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}

    with open(features) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(edges) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    print(label_map)
    temp = [[[k]*len(v), list(v)] for k,v in adj_lists.items()]
    temp2 = list(zip(*temp))
    src = list(itertools.chain.from_iterable(temp2[0]))
    dst = list(itertools.chain.from_iterable(temp2[1]))
    return torch.tensor(feat_data).float(), torch.tensor(labels).int().flatten(), (src, dst)


# In[6]:


def load_graphs(feat_data, labels, adj_lists):
    g = DGLGraph()
    g.add_nodes(feat_data.shape[0])
    g.ndata['feat'] = feat_data
    g.add_edges(adj_lists[0], adj_lists[1])
    g.ndata['labels'] = labels
    return g


# In[7]:


feat_data, labels, adj_lists = load_hate(os.path.join(data_path, 'hate/users_hate_all.content'), os.path.join(data_path, 'hate/users.edges'), 320)

annotated_idx = (labels != 1).nonzero().numpy().flatten()
train_idx, test_idx = train_test_split(annotated_idx, test_size=0.4,)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[torch.tensor(train_idx)]=True
val_mask[torch.tensor(test_idx)]=True

labels[labels == 1] = -1
labels[labels == 2] = 1
labels = labels.float()
graph = load_graphs(feat_data, labels, adj_lists)


# In[8]:


training_normals = (labels==0) * train_mask
training_hatefuls = (labels==1) * train_mask
ratio_h2n = training_hatefuls.sum().float() / training_normals.sum().float()

bernoulli = torch.distributions.bernoulli.Bernoulli(ratio_h2n)


# ## Model

# In[52]:


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()

        self.droplayer = nn.Dropout(p=dropout)

        # input layer
        self.inplayer = nn.Linear(in_feats, n_hidden)

        self.layers = nn.ModuleList()
        # hidden layers
        for i in range(n_layers):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))

        # output layer
        self.outlayer = nn.Linear(n_hidden, n_classes)

    def forward(self, features, graph):
        h = features
        h = self.inplayer(h)
        h = self.droplayer(h)

        for layer in self.layers:
            h = layer(graph, h)

        h = self.outlayer(h)
        return h


# ## Training loop

# In[21]:


def predict_logits(model, device, graph, mask=None):
    model.eval()
    with torch.no_grad():
        features = graph.ndata['feat'].to(device)
        logits = model(features, graph).flatten()

        if mask is not None:
            logits = logits[mask]
    return logits

def evaluate(logits, labels, mask=None):
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]

    sigLayer = nn.Sigmoid()
    predictions_scores = sigLayer(logits).detach().numpy()
    roc_auc = metrics.roc_auc_score(labels, predictions_scores)

    indices = (logits > 0).long()
    correct = torch.sum(indices == labels)
    return (roc_auc, correct.item() * 1.0 / len(labels))


# In[22]:


def evaluate_loss(model, criterion, device, val_mask, graph):
    model.eval()

    #validation phase
    with torch.set_grad_enabled(False):
        feat = graph.ndata['feat'].to(device)
        outputs = model(feat, graph).flatten()
        labels = graph.ndata['labels'].to(device)
        loss = criterion(outputs[val_mask], labels[val_mask])

    return loss.item()


# In[47]:


#Code for supervised training
def train_model(model, criterion, optimizer, scheduler, device, checkpoint_path, graph, checkpoint_iter, hyperparams, num_epochs=25):
    metrics_dict = {}
    metrics_dict["train"] = {}
    metrics_dict["valid"] = {}
    metrics_dict["train"]["loss"] = {}
    metrics_dict["train"]["loss"]["epochwise"] = []
    metrics_dict["valid"]["loss"] = {}
    metrics_dict["valid"]["loss"]["epochwise"] = []

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):

        und_sampled_normal_idx = training_normals.nonzero()[
            bernoulli.sample([training_normals.sum()]).bool()].flatten()

        balanced_train_mask = torch.zeros(train_mask.size(0),dtype=torch.bool)
        balanced_train_mask[training_hatefuls] = True
        balanced_train_mask[und_sampled_normal_idx] = True

        #train phase
        scheduler.step()
        model.train()
        optimizer.zero_grad()
        # forward
        # track history if only in train
        forward_start_time  = time.time()
        feats = graph.ndata['feat'].to(device)
        outputs = model(feats, graph).flatten()
        labels = graph.ndata['labels']
        labels = labels.to(device)
        loss = criterion(outputs[balanced_train_mask], labels[balanced_train_mask])
        epoch_loss = loss.item()
        loss.backward()
        optimizer.step()
        forward_time = time.time() - forward_start_time

        #validation phase
        val_epoch_loss = evaluate_loss(model, criterion, device, val_mask, graph)

        metrics_dict["train"]["loss"]["epochwise"].append(epoch_loss)
        metrics_dict["valid"]["loss"]["epochwise"].append(val_epoch_loss)

        # deep copy the model
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch%checkpoint_iter==0:
            print('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
            print('-' * 10)
            print('\n')
            print('Train Loss: {:.4f} \n'.format(epoch_loss))
            print('Validation Loss: {:.4f} \n'.format(val_epoch_loss))

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'full_metrics': metrics_dict,
            'hyperparams': hyperparams
            }, '%s/net_epoch_%d.pth' % (checkpoint_path, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f} \n'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

bestauc = 0
bestaccuracy = 0

for hiddim in hid_dims:
    for nlayer in n_layers:
        for drpout in dropouts:
            for wtdecay in wt_decays:
                # create GCN model
                model = GraphSAGE(graph.ndata['feat'].shape[1], hiddim, 1, nlayer, activation, drpout, aggregator_type)
                model.to(device)
                criterion = nn.BCEWithLogitsLoss()
                model_parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = optim.Adam(model_parameters, lr=learning_rate, weight_decay = wtdecay)
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=stpsize, gamma=0.1)
                hyper_params = {'hid_dim': hiddim,
                   'n_layers' : nlayer,
                   'dropout' : drpout,
                   'wt_decay' : wtdecay}
                new_out_path = os.path.join(out_path, str(hiddim) + '_' + str(nlayer) +'_' + str(drpout) + '_' + str(wtdecay))

                if not os.path.exists(new_out_path):
                    os.makedirs(new_out_path)

                bst_model = train_model(model, criterion, optimizer, exp_lr_scheduler, device, new_out_path, graph, checkpt_iter, hyper_params, n_epochs)
                logits = predict_logits(bst_model, device, graph, val_mask)
                auc, accuracy = evaluate(logits.cpu(), labels[val_mask].long())

                if auc > bestauc:
                    bestauc = auc
                    finalbstmodel = bst_model
                    bestaccuracy = accuracy

print("Best auc and accuracy: ", bestauc, bestaccuracy)
