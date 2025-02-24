import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

# ----- GCN Model for Node Selection -----
class GCNSelector(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        """
        A simple two–layer GCN that produces a scalar logit per node.
        The higher the logit, the more likely the node is to be selected.
        """
        super(GCNSelector, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)  # output a scalar score per node

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)  # shape: [num_nodes, 1]
        return x.squeeze(-1)          # shape: [num_nodes]