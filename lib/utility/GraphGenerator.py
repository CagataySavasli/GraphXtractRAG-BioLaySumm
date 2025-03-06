from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import torch


class GraphGenerator:
    def __init__(self):
        # Initialize the variables
        self.row = None
        self.nodes = None
        self.sentences = None
        self.header_clusters = []
        self.position_sentences = []

    def set_row(self, row: pd.Series):
        self.row = row
        self.nodes = [x for y in self.row['sections_embedding'] for x in y]
        self.sentences = [x for y in self.row['sections'] for x in y]
        self.header_clusters = self.row['heading_clusters']
        self.position_sentences = [x/len(self.sentences) for x in range(len(self.sentences))]

    def reset(self):
        self.nodes = None
        self.sentences = None
        self.header_clusters = []
        self.position_sentences = []

    def get_sentences(self):
        return self.sentences
    def create_graph(self):

        # Tüm cosine similarity değerlerini hesapla
        similarities = cosine_similarity(self.nodes)

        # Ortalama cosine similarity hesaplama
        avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])

        # Ortalama değerden büyük olan kenarları seç
        edges = [
            (i, j, similarities[i, j])
            for i in range(len(self.nodes)) for j in range(len(self.nodes))
            if i != j and similarities[i, j] > avg_similarity
        ]

        edges_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t().contiguous()

        if len(self.nodes[0]) == 768:
            for idx in range(len(self.nodes)):
                self.nodes[idx].insert(0, self.position_sentences[idx])
                self.nodes[idx].insert(0, self.header_clusters[idx])

        data = Data(x=torch.tensor(self.nodes, dtype=torch.float), edge_index=edges_index)

        return data

# %%
