import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv

class MIXSelector(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 1, dropout: float = 0.5):
        """
        İki katmanlı GAT tabanlı model. Çıktı olarak her düğüm için bir skalar logit üretir.
        """
        super(MIXSelector, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)  # Normalizasyon katmanı ekleniyor.
        self.dropout = nn.Dropout(dropout)  # Dropout katmanı.
        self.conv2 = GCNConv(hidden_channels * heads, 1)  # Her düğüm için skaler çıktı

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)  # Normalizasyon uygulandı.
        x = self.dropout(x)  # Dropout uygulaması.
        x = self.conv2(x, edge_index)  # shape: [num_nodes, 1]
        x = x.squeeze(-1)  # shape: [num_nodes]
        #x = torch.clamp(x, min=-50, max=50)  # Logit değerlerini sınırlıyoruz.
        return x
