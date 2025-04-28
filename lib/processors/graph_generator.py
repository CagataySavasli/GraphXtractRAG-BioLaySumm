from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Union


class GraphGenerator:
    def __init__(self):
        self.row = None
        self.nodes = None
        self.sentences = None
        self.header_clusters = []
        self.position_sentences = []

    def set_row(self, row: pd.Series):
        self.row = row
        # nodes: list of embedding vectors (list of floats)
        self.nodes = [x for y in row['sections_embedding'] for x in y]
        self.sentences = row['sentences']
        self.header_clusters = row['heading_clusters']
        self.position_sentences = [i / len(self.sentences) for i in range(len(self.sentences))]
        self.similarities = row['similarities']
        self.avg_similarity = row['avg_similarity']

    def reset(self):
        self.row = None
        self.nodes = None
        self.sentences = None
        self.header_clusters = []
        self.position_sentences = []

    def create_graph(self) -> Data:
        # 1) Edge listesini topla
        edge_list: List[tuple[int,int]] = []
        edge_attr_list: List[list[float]] = []
        N = len(self.nodes)
        for i in range(N):
            for j in range(N):
                if i != j and self.similarities[i, j] > self.avg_similarity:
                    edge_list.append((i, j))
                    # iki öznitelik: uzaklıkın tersi ve similarity
                    edge_attr_list.append([1.0/abs(i-j), float(self.similarities[i, j])])

        # 2) edge_index ve edge_attr tensor’larını oluştur
        if edge_list:
            src, dst = zip(*edge_list)
            edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
            edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)
        else:
            # sıfır kenarlı graph için
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr  = torch.empty((0, 2), dtype=torch.float)

        # 3) Header ve position bilgisini embedding listesine ekle
        #    (orijinal embed dim 768 ise)
        if self.nodes and len(self.nodes[0]) == 768:
            for idx, emb in enumerate(self.nodes):
                emb.insert(0, self.header_clusters[idx])
                emb.insert(1, self.position_sentences[idx])

        # 4) Node feature tensor’u
        x = torch.tensor(self.nodes, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def generate_from_row(self, row: pd.Series) -> Data:
        self.reset()
        self.set_row(row)
        return self.create_graph()

    __call__ = generate_from_row

    def generate_batch(
        self,
        rows: Union[pd.DataFrame, Iterable[pd.Series]],
        max_workers: int = None
    ) -> Batch:
        if isinstance(rows, pd.DataFrame):
            iterable = (rows.iloc[i] for i in range(len(rows)))
        else:
            iterable = rows

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            data_list: List[Data] = list(
                exe.map(lambda r: GraphGenerator().generate_from_row(r), iterable)
            )

        return Batch.from_data_list(data_list)
