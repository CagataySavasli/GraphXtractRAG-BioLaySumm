from src.rag_factories.GraphRAG_Factory import GraphRAG
from src.rag_factories.SimilarityRAG_Factory import RAG
from src.rag_factories.PersonalizationGraphRAG import PersonalizationGraphRAG
from src.rag_factories.GESRAG_Factory import GESRAG
import pandas as pd


class RAG_Factory:
    def __init__(self, case_builder, n, n_2=None, n_3=None):
        self.strategy = case_builder.rag_strategy
        self.case = case_builder.rag_case
        self.n = n
        self.n_2 = n_2
        self.n_3 = n_3

        self.factory = self.build_factory()

    def build_factory(self):
        if self.strategy == "similarityRAG":
            return RAG(self.case, self.n, self.n_2, self.n_3)
        elif self.strategy == "graphRAG":
            return GraphRAG(self.case, self.n)
        elif self.strategy == "personalizationGraphRAG":
            return PersonalizationGraphRAG(self.case, self.n)
        elif self.strategy == "GESRAG":
            return GESRAG(self.n)
        else:
            raise ValueError("Unsupported RAG strategy")

    def set_row(self, row: pd.Series):
        self.factory.set_row(row)

    def get_n_sentences(self):
        return self.factory.get_n_sentences()
