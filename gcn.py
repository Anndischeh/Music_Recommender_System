import torch
import torch.nn as nn
import numpy as np

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj_matrix):
        x = torch.relu(self.fc1(torch.matmul(adj_matrix, x)))
        x = self.fc2(torch.matmul(adj_matrix, x))
        return x
