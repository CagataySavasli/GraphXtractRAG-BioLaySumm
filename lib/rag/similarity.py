from typing import Any

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityRAG:

    def __init__(self, n = 10):
        self.n = n

    def run(self, row):
        sections_embedding = row['section_embeddings'][0]#[2:]
        section_sentences = row['sections'][0]#[2:]

        sections_embedding = [x for y in sections_embedding for x in y]
        title_embedding = row['title_embeddings'].tolist()
        sections_sentences = [x for y in section_sentences for x in y]



        avg_similarity, similarities = self.calculate_similarity(sections_embedding, title_embedding)
        top_n_index = self.get_top_n_index(similarities, self.n)
        top_n_sentences = self.get_top_n_sentences(top_n_index, sections_sentences)

        return top_n_sentences

    def calculate_similarity(self, sections_embeddings: list[Any], title_embedding: list[float]) -> tuple[float, np.ndarray]:
        # nodes = [x for y in row['sections_embedding'] for x in y]

        # Calculate all cosine similarities
        similarities = cosine_similarity(sections_embeddings, title_embedding)

        # Calculate avg cosine similarities
        avg_similarity = np.mean(similarities, dtype=np.float64)

        return avg_similarity, similarities

    def get_top_n_index(self, similarities, n: int) -> list[int]:

        combined = []
        for idx, similarity in enumerate(similarities):
            combined.append((idx, similarity))

        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)

        top_n = sorted_combined[:n]

        top_n_index = [item[0] for item in top_n]
        return top_n_index

    def get_top_n_sentences(self, top_n_index: list[int], sentences: list[Any]) -> list[Any]:

        top_n_sentences = []
        for idx in top_n_index:
            sentence = sentences[idx]
            top_n_sentences.append(sentence)

        return top_n_sentences