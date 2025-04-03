from lib.rag_factories.AbstractRAG_Factory import AbstractRAG_Factory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityRAG(AbstractRAG_Factory):

    def __init__(self, case, n):
        print("SimilarityRAG Factory")

        self.case = case
        self.n = n

        self.row = None

        self.nodes = None
        self.sentences = None

        self.top_bottom_checker = None

    def set_row(self, row):
        self.reset()
        self.row = row
        self.nodes = [x for y in row['sections_embedding'] for x in y]
        self.sentences = [x for y in row['sections'] for x in y]

    def reset(self):
        self.nodes = None
        self.sentences = None

    def calculate_similarities(self):
        references = np.array(self.row['title_embedding'], dtype=float)
        similarity = cosine_similarity([references], self.nodes).tolist()
        return similarity[0]

    def get_top_n_index(self, similarity):
        index = [x for x in range(len(similarity))]

        combined = list(zip(similarity, index))

        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=self.top_bottom_checker)

        top_n = sorted_combined[:self.n]
        top_n_sentences = [item[1] for item in top_n]

        return top_n_sentences

    def selector(self):
        if self.case == 'top':
            self.top_bottom_checker = True
        elif self.case == 'bottom':
            self.top_bottom_checker = False
        else:
            return None
    def get_n_sentences(self):
        self.selector()
        similarity = self.calculate_similarities()
        selected_indexes = self.get_top_n_index(similarity)

        return [self.sentences[i] for i in selected_indexes]




