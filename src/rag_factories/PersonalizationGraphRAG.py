from src.rag_factories.AbstractRAG_Factory import AbstractRAG_Factory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class PersonalizationGraphRAG(AbstractRAG_Factory):
    def __init__(self, case, n):
        self.n = n
        self.case = case
        self.row = None
        self.nodes = None
        self.sentences = None

        self.personalization = None

        self.G = None
        self.pagerank = None

        self.sorted_nodes = None

    def set_row(self, row):
        self.reset()
        self.row = row
        # Sections embedding'lerinden node'ları oluştur
        self.nodes = [x for y in row['sections_embedding'] for x in y]
        # Sections altındaki cümleleri al
        self.sentences = [x for y in row['sections'] for x in y]

        # Başlık, özet ve anahtar kelimelerin embedding'lerini personalization için kullan
        self.personalization = [
            list(row['title_embedding']),
            list(row['abstract_embedding']),
            list(row['keywords_embedding'])
        ]

    def reset(self):
        self.nodes = None
        self.sentences = None
        self.G = None
        self.pagerank = None
        self.sorted_nodes = None
        self.personalization = None

    def create_graph(self):
        # Cosine similarity hesapla
        similarities = cosine_similarity(self.nodes)

        # Ortalama cosine similarity hesapla
        avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])

        # Ortalama similarity'den büyük olan kenarları seç
        edges = [
            (i, j, similarities[i, j])
            for i in range(len(self.nodes)) for j in range(len(self.nodes))
            if i != j and similarities[i, j] > avg_similarity
        ]

        # Graf oluştur
        self.G = nx.Graph()
        for edge in edges:
            self.G.add_edge(edge[0], edge[1], weight=edge[2])

    def calculate_personalized_pagerank(self):
        # Ağırlıklar: Başlık, özet ve anahtar kelimeler için
        weights = {
            'title': 0.5,  # Başlığa daha fazla ağırlık
            'abstract': 0.3,  # Özet
            'keywords': 0.2  # Anahtar kelimeler
        }

        # Personalization vektörü oluştur
        personalization_vector = {}
        for i, node in enumerate(self.nodes):
            # Her bileşen için benzerlik hesapla ve ağırlıklı toplam al
            title_similarity = np.mean(cosine_similarity([node], [self.personalization[0]]))
            abstract_similarity = np.mean(cosine_similarity([node], [self.personalization[1]]))
            keywords_similarity = np.mean(
                cosine_similarity([node], [self.personalization[2]]))

            personalization_vector[i] = (
                    weights['title'] * title_similarity +
                    weights['abstract'] * abstract_similarity +
                    weights['keywords'] * keywords_similarity
            )

        # Normalize personalization vektörü
        total_weight = sum(personalization_vector.values())
        self.personalization = {k: v / total_weight for k, v in personalization_vector.items()}

        # Personalized PageRank hesapla
        self.pagerank = nx.pagerank(self.G, alpha=0.85, personalization=self.personalization)
        self.sorted_nodes = sorted(self.pagerank, key=self.pagerank.get, reverse=True)

    def top_n_nodes(self, N=None):
        if N is None: N = self.n
        return self.sorted_nodes[:N]

    def bottom_n_nodes(self, N=None):
        if N is None: N = self.n
        return self.sorted_nodes[-N:]

    def mixed_n_nodes(self):
        top_N = self.n // 2
        bottom_N = self.n - top_N
        return self.top_n_nodes(top_N) + self.bottom_n_nodes(bottom_N)

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
        self.calculate_personalized_pagerank()
        selected_indexes = self.selector()
        return [self.sentences[i] for i in selected_indexes]
