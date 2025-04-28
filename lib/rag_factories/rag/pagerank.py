from torch_geometric.data import Data
import networkx as nx
from lib.rag_factories.rag.abstract_rag import AbstractRAG_Factory
from lib.utility.case_builder import CaseBuilder

class PageRankRAG(AbstractRAG_Factory):
    def __init__(self):
        print("PageRankRAG Factory")
        self.case_builder = CaseBuilder()
        self.n = self.case_builder.rag_n

        self.row = None
        self.nodes = None
        self.sentences = None
        self.graph = None
        self.pagerank = None
        self.sorted_nodes = None

    def set_row(self, row, graph):
        self.reset()
        self.row = row
        self.nodes = [x for y in row['sections_embedding'] for x in y]
        self.sentences = row['sentences']
        self.graph = graph

    def reset(self):
        self.nodes = None
        self.sentences = None
        self.graph = None
        self.pagerank = None
        self.sorted_nodes = None

    def calculate_pagerank(self):
        # Eğer torch_geometric.data.Data objesi geldiyse, önce NetworkX grafına çevir
        if isinstance(self.graph, Data):
            data = self.graph
            G = nx.DiGraph()
            # Tüm düğümleri ekle
            G.add_nodes_from(range(data.num_nodes))
            # Kenarları, benzerlik (edge_attr[:,1]) ağırlığıyla ekle
            for idx in range(data.edge_index.size(1)):
                u = int(data.edge_index[0, idx])
                v = int(data.edge_index[1, idx])
                # edge_attr her zaman [inverse_distance, similarity] formatında
                sim = float(data.edge_attr[idx][1])
                G.add_edge(u, v, weight=sim)
        else:
            # Zaten bir NetworkX grafı gelmişse direkt kullan
            G = self.graph

        # PageRank hesapla (varsayılan weight='weight')
        self.pagerank = nx.pagerank(G, weight='weight')
        # Puanlara göre büyükten küçüğe düğüm indeksleri
        self.sorted_nodes = sorted(self.pagerank, key=self.pagerank.get, reverse=True)

    def top_n_nodes(self):
        return self.sorted_nodes[:self.n]

    def get_n_sentences(self):
        self.calculate_pagerank()
        selected_indexes = self.top_n_nodes()
        return [self.sentences[i] for i in selected_indexes]
