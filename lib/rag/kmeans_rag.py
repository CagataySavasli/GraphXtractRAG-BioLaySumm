import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from typing import Any


class KMeansRAG:
    def __init__(self, n=10):
        self.n = n

    def run(self, row):
        # 1. Verileri Hazırla
        sections_embedding = row['section_embeddings'][0]
        section_sentences = row['sections'][0]

        # Nested listeleri düzleştir (flatten)
        all_embeddings = [x for y in sections_embedding for x in y]
        all_sentences = [x for y in section_sentences for x in y]

        # Eğer cümle sayısı istenen N'den azsa hepsini döndür
        if len(all_embeddings) <= self.n:
            return all_sentences

        # 2. K-means Kümeleme
        # n_init='auto' ve random_state ile tutarlı sonuçlar alıyoruz
        kmeans = KMeans(n_clusters=self.n, n_init='auto', random_state=42)
        kmeans.fit(all_embeddings)

        # 3. Her küme merkezine en yakın cümleyi bul (Representativeness)
        # closest_indices, her bir cluster center'a en yakın olan noktanın indexini verir
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, all_embeddings)

        # Indexleri orijinal sırasına göre dizmek okuma akışı için daha iyidir
        sorted_indices = sorted(closest_indices)

        top_n_sentences = [all_sentences[idx] for idx in sorted_indices]

        return top_n_sentences