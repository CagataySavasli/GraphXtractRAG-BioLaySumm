from lib.rag_factories.SimilarityRAG_Factory import SimilarityRAG
from lib.rag_factories.PageRankRAG_Factory import PageRankRAG
from lib.rag_factories.PersonalizationPageRankRAG_Factory import PersonalizationPageRankRAG
from lib.rag_factories.GraphXtractRAG_Factory import GraphXtractRAG

from lib.utility.CaseBuilder import CaseBuilder

import pandas as pd


class RAG_Factory:
    def __init__(self):
        self.case_builder = CaseBuilder()
        self.strategy = self.case_builder.rag_strategy
        self.case = self.case_builder.rag_case
        self.n = self.case_builder.rag_n

        self.factory = self.build_factory()

    def build_factory(self):
        if self.strategy == "similarity":
            return SimilarityRAG(self.case, self.n)
        elif self.strategy == "pagerank":
            return PageRankRAG(self.case, self.n)
        elif self.strategy == "per_pagerank":
            return PersonalizationPageRankRAG(self.case, self.n)
        elif self.strategy == "graphxtract":
            return GraphXtractRAG(self.n)
        else:
            raise ValueError("Unsupported RAG strategy")

    def set_row(self, row: pd.Series):
        self.factory.set_row(row)

    def get_n_sentences(self):
        return self.factory.get_n_sentences()
