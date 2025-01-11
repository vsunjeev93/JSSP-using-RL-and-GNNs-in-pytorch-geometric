import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool


class GIN(torch.nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(n_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        MLP1_layers = [
            self.linear1,
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            self.linear2,
            nn.ReLU(),
        ]
        MLP2_layers = [
            self.linear3,
            nn.BatchNorm1d(hidden_dim),
            self.linear4,
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ]
        self.MLP1 = nn.Sequential(*MLP1_layers)
        self.MLP2 = nn.Sequential(*MLP2_layers)
        self.conv1 = GINConv(self.MLP1, eps=0)
        self.conv2 = GINConv(self.MLP2, eps=0)

    def forward(self, data, reverse=False):
        """
        data- batch graph data
        returns avg embedding (x_pool) and node embeddings (x)
        """
        if reverse:
            edge_index = data.reversed_edge_index
        else:
            edge_index = data.edge_index
        node_features = data.x
        batch = data.batch
        x = self.conv1(node_features, edge_index)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.dropout(x, training=self.training)

        ### graph node
        x_pool = global_mean_pool(x, batch)

        return x_pool, x
