"""GraphBased paper Step 3c (online): SimGNN adapted for tracklet matching.

Reuses the cloned SimGNN's AttentionModule + TensorNetworkModule (AIC21-MTMC/SimGNN),
but adapts the model for our task:
  - node features are continuous 2048-d ReID embeddings (not one-hot node labels),
    so the 1st GCN takes `input_dim` channels;
  - the output is a same/different-vehicle probability (trained with BCE), not a
    regressed graph-edit-distance similarity.

Given two tracklet graphs -> a similarity score in [0, 1] (1 = same vehicle).
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# reuse the user's cloned SimGNN layer implementations
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # ablation/simgnn -> repo root
sys.path.insert(0, os.path.join(REPO, 'SimGNN', 'src'))
from layers import AttentionModule, TenorNetworkModule  # noqa: E402


def default_args(filters=(128, 64, 32), tensor_neurons=16, bottle_neck=16,
                 bins=16, dropout=0.5, histogram=False):
    return SimpleNamespace(
        filters_1=filters[0], filters_2=filters[1], filters_3=filters[2],
        tensor_neurons=tensor_neurons, bottle_neck_neurons=bottle_neck,
        bins=bins, dropout=dropout, histogram=histogram)


class SimGNNMatcher(nn.Module):
    def __init__(self, input_dim=2048, args=None):
        super().__init__()
        self.args = args or default_args()
        self.feature_count = (self.args.tensor_neurons + self.args.bins
                              if self.args.histogram else self.args.tensor_neurons)
        self.conv1 = GCNConv(input_dim, self.args.filters_1)
        self.conv2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.conv3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fc1 = nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring = nn.Linear(self.args.bottle_neck_neurons, 1)

    def conv_pass(self, edge_index, x):
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.args.dropout, training=self.training)
        x = F.dropout(F.relu(self.conv2(x, edge_index)), p=self.args.dropout, training=self.training)
        return self.conv3(x, edge_index)

    def histogram(self, a, b):
        s = torch.mm(a, b.t()).detach().view(-1, 1)
        h = torch.histc(s, bins=self.args.bins)
        return (h / h.sum()).view(1, -1)

    def forward(self, ei1, x1, ei2, x2):
        f1 = self.conv_pass(ei1, x1)
        f2 = self.conv_pass(ei2, x2)
        pooled1 = self.attention(f1)
        pooled2 = self.attention(f2)
        scores = torch.t(self.tensor_network(pooled1, pooled2))
        if self.args.histogram:
            scores = torch.cat((scores, self.histogram(f1, f2)), dim=1).view(1, -1)
        scores = F.relu(self.fc1(scores))
        return torch.sigmoid(self.scoring(scores)).view(-1)  # (1,)
