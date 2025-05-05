from torch_geometric.data import Data
import networkx as nx
from lib.rag_factories.rag.abstract_rag import AbstractRAG_Factory
from lib.utility.case_builder import CaseBuilder


class PageRankRAG(AbstractRAG_Factory):
    def __init__(self):

        self.case_builder = CaseBuilder()
        self.n = self.case_builder.rag_n

        self.row = None
        self.sentences = None
        self.graph = None
        self.pagerank = None
        self.sorted_nodes = None

    def set_row(self, row, graph):
        self.reset()
        self.row = row
        self.sentences = row['sentences']
        self.graph = graph

    def reset(self):
        self.row = None
        self.sentences = None
        self.graph = None
        self.pagerank = None
        self.sorted_nodes = None

    def calculate_pagerank(self):
        # 1) PyG Data objesi geldiyse, önce NetworkX grafına dönüştür
        if isinstance(self.graph, Data):
            data = self.graph
            num_sent = len(self.sentences)

            G = nx.DiGraph()
            # sadece gerçek sentence sayısı kadar node ekle
            G.add_nodes_from(range(num_sent))

            # kenarları ekle; U/V mutlaka [0, num_sent) aralığında olmalı
            for idx in range(data.edge_index.size(1)):
                u = int(data.edge_index[0, idx])
                v = int(data.edge_index[1, idx])
                if u >= num_sent or v >= num_sent:
                    continue
                # edge_attr = [inverse_distance, similarity]
                sim = float(data.edge_attr[idx][1])
                G.add_edge(u, v, weight=sim)
        else:
            # önceden networkx grafı gelmişse direkt kullan
            G = self.graph

        # 2) PageRank hesapla
        self.pagerank = nx.pagerank(G, weight='weight')
        # 3) en yüksekten en düşüğe düğüm indekslerini sırala
        self.sorted_nodes = sorted(self.pagerank, key=self.pagerank.get, reverse=True)

    def top_n_nodes(self):
        # olası OOB durumları temizle ve sadece ilk self.n elemanı al
        valid = [i for i in self.sorted_nodes if 0 <= i < len(self.sentences)]
        return valid[:self.n]

    def get_n_sentences(self):
        self.calculate_pagerank()
        selected = self.top_n_nodes()
        # artık kesinlikle geçersiz indeks yok
        return [self.sentences[i] for i in selected]
