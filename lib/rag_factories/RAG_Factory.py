from lib.rag_factories.GraphRAG_Factory import GraphRAG
from lib.rag_factories.SimilarityRAG_Factory import RAG
from lib.rag_factories.PersonalizationGraphRAG import PersonalizationGraphRAG

from lib.utility.CaseBuilder import CaseBuilder

from lib.rag_factories.GESRAG_Factory import GESRAG
import pandas as pd


class RAG_Factory:
    def __init__(self):
        self.case_builder = CaseBuilder()
        self.strategy = self.case_builder.rag_strategy
        self.case = self.case_builder.rag_case
        self.n = self.case_builder.rag_n

        self.factory = self.build_factory()

    def build_factory(self):
        if self.strategy == "similarityRAG":
            return RAG(self.case, self.n)
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
