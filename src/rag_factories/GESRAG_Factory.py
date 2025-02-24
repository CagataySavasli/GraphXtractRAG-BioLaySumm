from src.rag_factories.AbstractRAG_Factory import AbstractRAG_Factory
from src.selectors.GCNSelector import GCNSelector
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
class GESRAG(AbstractRAG_Factory):
    def __init__(self, n):
        self.model = GCNSelector(768, 128)
        self.model.load_state_dict(torch.load("/Users/cagatay/Desktop/CS/Projects/BioLaySumm-BiOzU/models/GCN_20_selector.pth", weights_only=True))
        self.model.eval()

        self.n =n

        self.row = None
        self.nodes = None
        self.sentences = None

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

        data = Data(x=torch.tensor(self.nodes, dtype=torch.float), edge_index=edges_index)

        return data

    def set_row(self, row: pd.Series):
        self.row = row
        self.nodes = [x for y in self.row['sections_embedding'] for x in y]
        self.sentences = [x for y in self.row['sections'] for x in y]

    def select_sentences(self, data):

        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)  # shape: [num_nodes]

            # Adım 2: Logit'leri softmax ile olasılığa dönüştürüyoruz.
            probs = F.softmax(logits, dim=0)  # shape: [num_nodes]

            # Adım 3: Olasılık dağılımından n düğümü örnekliyoruz.
            node_indices = torch.multinomial(probs, self.n, replacement=False)

            # Adım 4: Seçilen düğümlerin log olasılıklarının toplamı hesaplanır.
            selected_log_probs = torch.log(probs[node_indices]).sum()

            # Adım 5: Seçilen düğümlere ait cümleleri birleştirip prompt oluşturuyoruz.
            selected_sentences = [self.sentences[i] for i in node_indices.tolist()]

        return selected_sentences

    def get_n_sentences(self):
        data = self.create_graph()
        return self.select_sentences(data)