import torch
from torch_geometric.data import Batch
from lib.data.graph_generator import GraphGenerator


class GraphBatchCollator:
    def __init__(self, device):
        self.graph_generator = GraphGenerator()
        self.device = device

    def __call__(self, batch_list):
        """
        DataLoader tarafından çağrılan fonksiyon.
        batch_list: DatabaseConnector'dan gelen [row1, row2, ...] listesi.
        """
        graphs = []
        metadata = {
            "titles": [],
            "abstracts": [],
            "summaries": [],
            "sections": []
        }

        # Batch içindeki her bir satır için işlem yap
        for row in batch_list:
            # 1. Grafiği oluştur
            graph = self.graph_generator.generate_from_row(row)
            graphs.append(graph)

            # 2. Metin verilerini listele (Text verileri Tensor olmaz, liste tutulur)
            # Not: DatabaseConnector dataframe döndürüyorsa .values[0] ile, dict ise direkt alıyoruz.
            # Kodunuzda DataFrame döndüğü varsayılmıştır.

            # Helper function to ensure string
            def to_str(x): return " ".join(x) if isinstance(x, list) else str(x)

            metadata["titles"].append(to_str(row['title'].values[0]))
            metadata["abstracts"].append(to_str(row['abstract'].values[0]))
            metadata["summaries"].append(to_str(row['summary'].values[0]))

            # Cümleleri düzleştir (Flatten sections)
            flat_sentences = [s for section in row['sections'][0] for s in section]
            metadata["sections"].append(flat_sentences)

        # 3. PyG Batch Objesi Oluştur
        batch_graph = Batch.from_data_list(graphs)

        return batch_graph, metadata