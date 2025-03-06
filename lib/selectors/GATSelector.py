import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


# ----- GNN Model for Node Selection using Graph Attention Network (GAT) -----
class GATSelector(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 1):
        """
        A simple two-layer GAT that produces a scalar logit per node.
        The higher the logit, the more likely the node is to be selected.
        """
        super(GATSelector, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, 1, heads=1, concat=False)  # Output a scalar score per node

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)  # shape: [num_nodes, 1]
        return x.squeeze(-1)  # shape: [num_nodes]
