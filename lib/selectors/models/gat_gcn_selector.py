import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv

class GATGCNSelector(nn.Module):
    def __init__(self, in_channels: int = 770, hidden_channels: int = 128, heads: int = 1, dropout: float = 0.5):
        """
        İki katmanlı GAT tabanlı model. Çıktı olarak her düğüm için bir skalar logit üretir.
        """
        super(GATGCNSelector, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)  # Normalizasyon katmanı ekleniyor.
        self.dropout = nn.Dropout(dropout)  # Dropout katmanı.
        self.conv2 = GCNConv(hidden_channels * heads, 1)  # Her düğüm için skaler çıktı

    def forward(self, graph):
        x, edge_index, batch_idx = graph.x, graph.edge_index, graph.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index).squeeze(-1)     # [total_nodes]

        return x
        #
        # # Her graf için node logit’lerinin ortalamasını alıyoruz:
        # graph_logits = global_mean_pool(x.unsqueeze(-1), batch_idx)
        # # → [batch_size, 1]
        # return graph_logits.squeeze(-1)
