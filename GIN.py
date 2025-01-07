import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import group_argsort, index_to_mask, unbatch, softmax




class GIN(torch.nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(n_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        MLP1_layers = [self.linear1, nn.ReLU(), self.linear2, nn.ReLU()]
        MLP2_layers = [self.linear3, nn.ReLU(), self.linear4, nn.ReLU()]
        self.MLP1 = nn.Sequential(*MLP1_layers)
        self.MLP2 = nn.Sequential(*MLP2_layers)
        self.conv1 = GINConv(self.MLP1, eps=0)
        self.conv2 = GINConv(self.MLP2, eps=0)

    def forward(self, data,reverse=False, inf=10**10):
        if reverse:
            edge_index=data.reversed_edge_index
        else:
            edge_index=data.edge_index
        node_features = data.x
        batch = data.batch
        x = self.conv1(node_features, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)

        # mask=~mask
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)

        ### graph node
        x_pool = global_mean_pool(x, batch)

        return x_pool,x