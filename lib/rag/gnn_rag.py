import torch
import os
import pandas as pd
import numpy as np
from lib.data.graph_generator import GraphGenerator
from lib.model.selector.gnn_selector import GNNSelector


class GNNRAG:
    """
    Eğitilmiş GNN modelini yükler ve verilen veri satırı (row) için
    en önemli cümleleri seçer.
    """

    def __init__(self, model_path, hidden_dim=64, top_n=10, device=None):
        self.model_path = model_path
        self.hidden_dim = hidden_dim
        self.top_n = top_n
        self.graph_generator = GraphGenerator()

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Modeli mimarisini kurar ve ağırlıkları yükler."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")

        # GNN input dimension'ı belirlemek için varsayılan değer (768 embedding + 1 pos)
        # Ancak en garantisi bir dummy graph oluşturup bakmaktır,
        # şimdilik standart 769 kabul ediyoruz.
        input_dim = 769

        self.model = GNNSelector(in_channels=input_dim, hidden_channels=self.hidden_dim)

        # Ağırlıkları yükle
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # Inference modu
        print(f"GNN Model loaded from {self.model_path}")

    def select_sentences(self, row: pd.Series) -> list[str]:
        """
        Tek bir satır (makale) için GNN skorlarına göre cümle seçimi yapar.
        """
        try:
            # 1. Grafiği oluştur
            graph = self.graph_generator.generate_from_row(row).to(self.device)

            # 2. Modelden geçir (Forward Pass)
            with torch.no_grad():
                probs = self.model(graph)  # Shape: [Num_Nodes, 1]

            # 3. Cümle listesini al
            sentences_list = [s for section in row['sections'] for s in section]

            # 4. Top-N Seçimi
            num_nodes = probs.shape[0]
            k = min(self.top_n, num_nodes)

            if k == 0:
                return []

            # En yüksek skorlu k cümleyi bul
            top_k_scores, top_k_indices = torch.topk(probs.squeeze(), k=k)

            # Tensorden listeye çevir
            selected_indices_list = top_k_indices.detach().cpu().numpy().tolist()

            # İndeksleri cümlelere eşle
            selected_sentences = []
            for idx in selected_indices_list:
                if idx < len(sentences_list):
                    selected_sentences.append(sentences_list[idx])

            return selected_sentences

        except Exception as e:
            print(f"Error in GNN selection: {e}")
            return []

    # Standart arayüz uyumu için run metodu
    def run(self, row):
        return self.select_sentences(row)