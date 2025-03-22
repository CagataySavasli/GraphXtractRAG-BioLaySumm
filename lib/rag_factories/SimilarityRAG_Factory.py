from lib.rag_factories.AbstractRAG_Factory import AbstractRAG_Factory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityRAG(AbstractRAG_Factory):

    def __init__(self, case, title_n, abstract_n = 0, keywords_n = 0):
        print("SimilarityRAG Factory")
        self.case = case
        self.title_n = title_n
        self.abstract_n = abstract_n
        self.keywords_n = keywords_n

        self.n = self.title_n + self.abstract_n + self.keywords_n

        self.row = None

        self.nodes = None
        self.sentences = None

        self.ref_similarity = None
        self.selected_indexes = []

        self.top_bottom_checker = None

    def set_row(self, row):
        self.reset()
        self.row = row
        self.nodes = [x for y in row['sections_embedding'] for x in y]
        self.sentences = [x for y in row['sections'] for x in y]

    def reset(self):
        self.selected_indexes = []
        self.nodes = None
        self.sentences = None
        self.ref_similarity = None

    def calculate_similarities(self):

        referances = [np.array(ref) for ref in
                      self.row[['title_embedding', 'abstract_embedding', 'keywords_embedding']]]
        similarity = cosine_similarity(
            referances,
            self.nodes,
        ).tolist()

        self.ref_similarity = {
            'title': similarity[0],
            'abstract': similarity[1],
            'keywords': similarity[2]
        }

    def get_top_n_index(self, similarity):
        index = [x for x in range(len(similarity))]

        combined = list(zip(similarity, index))

        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=self.top_bottom_checker)

        top_n = sorted_combined[:self.n]
        top_n_sentences = [item[1] for item in top_n]

        return top_n_sentences

    def iter_selecte_list(self, index, N):
        n_goal = len(self.selected_indexes) + N

        i = 0
        while len(self.selected_indexes) < n_goal:
            if not index[i] in self.selected_indexes:
                self.selected_indexes.append(index[i])
            i += 1

    def selecte_n_index(self):

        title_top_n_index = self.get_top_n_index(self.ref_similarity['title'])
        abstract_top_n_index = self.get_top_n_index(self.ref_similarity['abstract'])
        keywords_top_n_index = self.get_top_n_index(self.ref_similarity['keywords'])

        self.iter_selecte_list(title_top_n_index, self.title_n, )
        self.iter_selecte_list(abstract_top_n_index, self.abstract_n, )
        self.iter_selecte_list(keywords_top_n_index, self.keywords_n, )

    def selector(self):
        if self.case == 'top':
            self.top_bottom_checker = True
        elif self.case == 'bottom':
            self.top_bottom_checker = False
        else:
            return None
    def get_n_sentences(self):
        self.selector()
        self.calculate_similarities()
        self.selecte_n_index()

        return [self.sentences[i] for i in self.selected_indexes]




