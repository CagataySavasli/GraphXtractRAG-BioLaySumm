from concurrent.futures import ThreadPoolExecutor
from torch_geometric.data import Batch
from lib.rag_factories.rag import SimilarityRAG, PageRankRAG, PersonalizationPageRankRAG, GraphXtractRAG
from lib.utility.case_builder import CaseBuilder
import pandas as pd
from typing import Iterable, List, Union

class RAG_Factory:
    def __init__(self):
        self.case_builder = CaseBuilder()
        self.strategy = self.case_builder.rag_strategy
        self.factory = self.build_factory()

    def build_factory(self):
        if self.strategy == "similarity":
            return SimilarityRAG()
        elif self.strategy == "pagerank":
            return PageRankRAG()
        elif self.strategy == "per_pagerank":
            return PersonalizationPageRankRAG()
        elif self.strategy == "graphxtract":
            return GraphXtractRAG()
        else:
            raise ValueError(f"Unsupported RAG strategy: {self.strategy}")

    def _process_single(self, row, graph, selector_model=None):
        # Thread-safe: her işlemde yeni factory üret
        local_factory = self.build_factory()

        if selector_model is not None:
            local_factory.model = selector_model  # Model override

        if self.strategy == "similarity":
            local_factory.set_row(row)
        else:
            local_factory.set_row(row, graph)

        return local_factory.get_n_sentences()

    def get_n_sentences_batch(
        self,
        row_batch: Union[pd.DataFrame, Iterable[pd.Series]],
        graph_batch: Union[Batch, Iterable] = None,
        selector_model=None,
        max_workers: int = None
    ) -> List[List[str]]:

        if isinstance(row_batch, pd.DataFrame):
            rows = [row_batch.iloc[i] for i in range(len(row_batch))]
        else:
            rows = list(row_batch)

        if self.strategy == "similarity" or graph_batch is None:
            graphs = [None] * len(rows)
        elif isinstance(graph_batch, Batch):
            graphs = graph_batch.to_data_list()
        else:
            graphs = list(graph_batch)

        if len(graphs) != len(rows):
            raise ValueError("row_batch ve graph_batch must have same length")

        # Thread-safe map with model injection
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            results = list(
                exe.map(
                    lambda args: self._process_single(*args, selector_model),
                    zip(rows, graphs)
                )
            )
        return results

    def get_n_sentences(self, row, graph=None):
        return self._process_single(row, graph)

    __call__ = get_n_sentences_batch