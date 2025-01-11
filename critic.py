import torch
from GIN import GIN


class critic(torch.nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.GIN = GIN(n_features, hidden_dim)
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, _ = self.GIN(data)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.output(x).squeeze()
        return x
