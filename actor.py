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



class actor(torch.nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.GIN1=GIN(n_features,hidden_dim)
        self.GIN2=GIN(n_features,hidden_dim)
        self.linear1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim+n_features, hidden_dim)
        self.action = nn.Linear(hidden_dim, 1)
        action_layers1 = [self.linear1, nn.ReLU()]
        self.action_MLP1 = nn.Sequential(*action_layers1)
        action_layers2 = [self.linear2, nn.ReLU(), self.action]
        self.action_MLP2 = nn.Sequential(*action_layers2)
        self.bn1=nn.BatchNorm1d(hidden_dim)

    def forward(self, data, inf=10**10):
        batch = data.batch
        mask=data.mask
        x_pool,x=self.GIN1(data)
        x_pool_b,x_b=self.GIN2(data,reverse=True)
        batch_index = unbatch(batch, batch)
        batch_index = torch.tensor([len(i) for i in batch_index])
        graph_embedding = torch.repeat_interleave(x_pool.to('cpu'), batch_index, dim=0).to('mps')
        #reversed edge avg node embedding
        graph_embedding_b=torch.repeat_interleave(x_pool_b.to('cpu'), batch_index, dim=0).to('mps')
        # final embedding before passing to the action layers has a skip connection from input features.
        final_embed = torch.cat((x, graph_embedding,x_b,graph_embedding_b), dim=1)
        x = self.action_MLP1(final_embed)
        x=torch.cat((x,data.x),dim=1)
        x=self.action_MLP2(x).squeeze()
        mask = torch.where(mask, 0, inf)
        x = x - mask.squeeze()
        x = softmax(x, index=batch)
        x = torch.stack(unbatch(x, batch))
        sample = torch.multinomial(x, num_samples=1).to('mps')
        log_actions=torch.log(torch.gather(x,index=sample,dim=1)).squeeze()
        sample = sample.squeeze() + data.graph_id_offset
        
        return sample, log_actions