import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityPreprocessor:

    def get_similarity(self, row: pd.Series) -> tuple[float, np.ndarray]:
        nodes = [x for y in row['sections_embedding'][0] for x in y]

        # Tüm cosine similarity değerlerini hesapla
        similarities = cosine_similarity(nodes)

        # Ortalama cosine similarity hesaplama
        avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])

        return avg_similarity, similarities

    __call__ = get_similarity