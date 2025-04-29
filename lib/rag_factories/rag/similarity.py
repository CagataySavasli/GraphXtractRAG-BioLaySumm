from lib.rag_factories.rag.abstract_rag import AbstractRAG_Factory
from lib.utility.case_builder import CaseBuilder

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityRAG(AbstractRAG_Factory):

    def __init__(self):
        print("SimilarityRAG Factory")

        self.case_builder = CaseBuilder()
        self.n = self.case_builder.rag_n


        self.row = None

        self.nodes = None
        self.sentences = None

        self.top_bottom_checker = True

    def set_row(self, row):
        self.reset()
        self.row = row
        self.nodes = [x for y in row['sections_embedding'] for x in y]
        self.sentences = [x for x in row['sentences']]

        # Verify data consistency
        if len(self.nodes) != len(self.sentences):
            raise ValueError(
                f"Mismatch between number of embeddings ({len(self.nodes)}) and sentences ({len(self.sentences)})")


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

    def get_n_sentences(self):
        similarity = self.calculate_similarities()
        selected_indexes = self.get_top_n_index(similarity)

        return [self.sentences[i] for i in selected_indexes if i < len(self.sentences)]

    # def get_n_sentences(self):
    #     if not self.nodes or not self.sentences:
    #         return []
    #
    #     similarity = self.calculate_similarities()
    #     selected_indexes = self.get_top_n_index(similarity)
    #
    #     # Ensure indices are within bounds
    #     valid_indexes = [i for i in selected_indexes if i < len(self.sentences)]
    #     return [self.sentences[i] for i in valid_indexes]



