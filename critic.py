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
from GIN import GIN



class critic(torch.nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.GIN=GIN(n_features,hidden_dim)
        self.linear1=torch.nn.Linear(hidden_dim,hidden_dim)
        self.linear2=torch.nn.Linear(hidden_dim,hidden_dim)
        self.output=torch.nn.Linear(hidden_dim,1)

    def forward(self, data):
        x,_=self.GIN(data)
        # x=torch.cat((x,dß),dim=1)
        x=self.linear1(x)
        x = torch.nn.functional.relu(x)
        x=self.linear2(x)
        x=torch.nn.functional.relu(x)
        x=self.output(x).squeeze()
        return x