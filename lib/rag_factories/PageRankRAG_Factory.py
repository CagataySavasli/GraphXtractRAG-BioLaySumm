from lib.rag_factories.AbstractRAG_Factory import AbstractRAG_Factory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class PageRankRAG(AbstractRAG_Factory):
    def __init__(self, case, n):
        print("PageRankRAG Factory")

        self.n = n
        self.case = case
        self.row = None
        self.nodes = None
        self.sentences = None

        self.G = None
        self.pagerank = None

        self.sorted_nodes = None

    def set_row(self, row):
        self.reset()
        self.row = row
        self.nodes = [x for y in row['sections_embedding'] for x in y]
        self.sentences = [x for y in row['sections'] for x in y]

    def reset(self):
        self.nodes = None
        self.sentences = None
        self.G = None
        self.pagerank = None
        self.sorted_nodes = None
    def calculate_pagerank(self):
        self.pagerank = nx.pagerank(self.G)
        self.sorted_nodes = sorted(self.pagerank, key=self.pagerank.get, reverse=True)

    def create_graph(self):
        labels = [i for i in range(len(self.nodes))]

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

        # Ağı oluşturmak (Graph)
        self.G = nx.Graph()
        for edge in edges:
            self.G.add_edge(edge[0], edge[1], weight=edge[2])

        # Kullanılan node'lar (aktif node'lar)
        active_nodes = set(node for edge in edges for node in edge[:2])

        # PageRank hesaplama
        pagerank = nx.pagerank(self.G)

        # En yüksek PageRank değerlerine sahip 5 node
        top_n_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:self.n]

        return top_n_nodes

    def top_n_nodes(self, N=None):
        if N is None: N = self.n
        return self.sorted_nodes[:N]

    def bottom_n_nodes(self, N=None):
        if N is None: N = self.n
        return self.sorted_nodes[-N:]

    def mixed_n_nodes(self):
        top_N = self.n // 2
        bottom_N = self.n - top_N
        return self.top_n_nodes(top_N).extend(self.bottom_n_nodes(bottom_N))

    def selector(self):
        if self.case == 'top':
            return self.top_n_nodes()
        elif self.case == 'bottom':
            return self.bottom_n_nodes()
        elif self.case == 'mixed':
            return self.mixed_n_nodes()
        else:
            return None

    def get_n_sentences(self):
        self.create_graph()
        self.calculate_pagerank()
        selected_indexes = self.selector()
        return [self.sentences[i] for i in selected_indexes]
