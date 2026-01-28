import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm


class GNNSelector(torch.nn.Module):
    """
    Graph Attention Network (GAT) based Sentence Selector.
    Outputs probabilities for each sentence node.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 1, heads: int = 2):
        super(GNNSelector, self).__init__()

        # Layer 1
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.4)
        self.bn1 = BatchNorm(hidden_channels * heads)

        # Layer 2
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.4)
        self.bn2 = BatchNorm(hidden_channels)

        # Classifier
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)

        # Prediction
        out = self.classifier(x)

        # Sigmoid for probability (0-1)
        return torch.sigmoid(out)