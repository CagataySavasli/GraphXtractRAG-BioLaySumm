# import pandas as pd
# import numpy as np
# import torch
# from torch_geometric.data import Data, Batch
# from concurrent.futures import ThreadPoolExecutor
# from typing import Iterable, List, Union, Dict, Any
#
#
# class GraphGenerator:
#     """
#     Generates a PyTorch Geometric Data object (graph) from a single row
#     of preprocessed article data.
#
#     This version is updated to work with datasets that DO NOT contain
#     'heading_clusters'. It uses sentence embeddings and positional info.
#     """
#
#     def __init__(self):
#         """Initializes the GraphGenerator."""
#         pass
#
#     def _extract_data_from_row(
#             self, row_data: Union[pd.Series, pd.DataFrame]
#     ) -> Dict[str, Any]:
#         """
#         Extracts, validates, and flattens data from the input row.
#         """
#
#         # Standardize input to pd.Series
#         if isinstance(row_data, pd.DataFrame):
#             if len(row_data) == 1:
#                 row_series = row_data.iloc[0]
#             else:
#                 raise ValueError(
#                     "Input DataFrame must contain exactly one row."
#                 )
#         elif isinstance(row_data, pd.Series):
#             row_series = row_data
#         else:
#             raise TypeError(
#                 "Input must be a pd.Series or a 1-row pd.DataFrame."
#             )
#
#         # 1. Extract node embeddings (sentence embeddings)
#         # Assumes 'section_embeddings' is list[list[list[float]]]
#         node_embeddings = [
#             embedding
#             for section in row_series['section_embeddings']
#             for embedding in section
#         ]
#
#         # 2. Extract sentences (text)
#         # Assumes 'sections' is list[list[str]]
#         sentences = [
#             sentence
#             for section in row_series['sections']
#             for sentence in section
#         ]
#
#         # 3. (REMOVED) Extract header cluster info
#         # header_clusters = row_series['heading_clusters']
#         # Yeni veride bu alan yok.
#
#         # 4. Calculate positional encodings for sentences
#         num_sentences = len(sentences)
#         position_sentences = [
#             i / num_sentences for i in range(num_sentences)
#         ] if num_sentences > 0 else []
#
#         # 5. Extract similarity matrix
#         similarities_data = row_series['similarities']
#
#         # Handle cases where similarities might be wrapped in a list
#         if isinstance(similarities_data, list) and len(similarities_data) > 0:
#             similarities_matrix = similarities_data[0]
#         else:
#             similarities_matrix = similarities_data
#
#         # Basic validation for similarities matrix
#         if not isinstance(similarities_matrix, np.ndarray):
#             # Try to convert if it's a list of lists (fallback)
#             if isinstance(similarities_matrix, list):
#                 similarities_matrix = np.array(similarities_matrix)
#             else:
#                 # If still not valid, this might raise an error downstream or here
#                 # For now, we assume it's fixable or valid
#                 pass
#
#         # 6. Extract average similarity
#         avg_similarity = row_series['avg_similarity']
#
#         return {
#             "nodes": node_embeddings,
#             "sentences": sentences,
#             # "header_clusters": header_clusters, # Removed
#             "position_sentences": position_sentences,
#             "similarities": similarities_matrix,
#             "avg_similarity": avg_similarity,
#         }
#
#     def _create_graph_from_data(self, data: Dict[str, Any]) -> Data:
#         """
#         Creates the graph Data object from the extracted data dictionary.
#         """
#         nodes = data['nodes']
#         similarities = data['similarities']
#         avg_similarity = data['avg_similarity']
#         # header_clusters = data['header_clusters'] # Removed
#         position_sentences = data['position_sentences']
#
#         # 1) Collect edge list based on similarity threshold
#         edge_list: List[tuple[int, int]] = []
#         edge_attr_list: List[list[float]] = []
#         num_nodes = len(nodes)
#
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 # Using (i, j) on numpy array
#                 if i != j and similarities[i, j] > avg_similarity:
#                     edge_list.append((i, j))
#                     # Edge features: inverse distance and similarity score
#                     edge_attr_list.append(
#                         [1.0 / abs(i - j), float(similarities[i, j])]
#                     )
#
#         # 2) Create edge_index and edge_attr tensors
#         if edge_list:
#             src, dst = zip(*edge_list)
#             edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
#             edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
#         else:
#             # Handle graph with zero edges
#             edge_index = torch.empty((2, 0), dtype=torch.long)
#             edge_attr = torch.empty((0, 2), dtype=torch.float)
#
#         # 3) Prepend position info to node features
#         #    We create a copy to avoid modifying the input data list.
#         node_features = [emb[:] for emb in nodes]  # Deep copy lists
#
#         if node_features:
#             for idx, emb in enumerate(node_features):
#                 # Eski kodda: emb.insert(0, header_clusters[idx])
#                 # Yeni kodda: Sadece position bilgisini ekliyoruz.
#                 emb.insert(0, position_sentences[idx])
#
#         # 4) Create node feature tensor
#         x = torch.tensor(node_features, dtype=torch.float)
#
#         return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
#
#     def generate_from_row(
#             self, row_data: Union[pd.DataFrame, pd.Series]
#     ) -> Data:
#         """
#         Main entry point. Generates a graph from a single row of data.
#         """
#         extracted_data = self._extract_data_from_row(row_data)
#         return self._create_graph_from_data(extracted_data)
#
#     # Alias __call__ to the main generation method
#     __call__ = generate_from_row
#
#     def generate_batch(
#             self,
#             rows: Union[pd.DataFrame, Iterable[pd.Series]],
#             max_workers: int = None
#     ) -> Batch:
#         """
#         Generates a Batch object from multiple rows using parallel processing.
#         """
#         if isinstance(rows, pd.DataFrame):
#             iterable = (rows.iloc[i] for i in range(len(rows)))
#         else:
#             iterable = rows
#
#         def process_row(row_data):
#             return GraphGenerator().generate_from_row(row_data)
#
#         with ThreadPoolExecutor(max_workers=max_workers) as exe:
#             data_list: List[Data] = list(
#                 exe.map(process_row, iterable)
#             )
#
#         return Batch.from_data_list(data_list)

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Union, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity  # Similarity hesabı için eklendi


class GraphGenerator:
    """
    Generates a PyTorch Geometric Data object (graph) from a single row
    of preprocessed article data.

    Updated:
    1. Removes dependency on 'heading_clusters'.
    2. Removes dependency on pre-calculated 'similarities' and 'avg_similarity' columns.
       It calculates the cosine similarity matrix from sentence embeddings on the fly.
    """

    def __init__(self):
        """Initializes the GraphGenerator."""
        pass

    def _extract_data_from_row(
            self, row_data: Union[pd.Series, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Extracts data and computes similarity matrix from embeddings.
        """

        # Standardize input to pd.Series
        if isinstance(row_data, pd.DataFrame):
            if len(row_data) == 1:
                row_series = row_data.iloc[0]
            else:
                raise ValueError(
                    "Input DataFrame must contain exactly one row."
                )
        elif isinstance(row_data, pd.Series):
            row_series = row_data
        else:
            raise TypeError(
                "Input must be a pd.Series or a 1-row pd.DataFrame."
            )

        # 1. Extract node embeddings (sentence embeddings)
        # Assumes 'section_embeddings' is list[list[list[float]]]
        node_embeddings = [
            embedding
            for section in row_series['section_embeddings']
            for embedding in section
        ]

        # 2. Extract sentences (text)
        # Assumes 'sections' is list[list[str]]
        sentences = [
            sentence
            for section in row_series['sections']
            for sentence in section
        ]

        # 3. Calculate positional encodings for sentences
        num_sentences = len(sentences)
        position_sentences = [
            i / num_sentences for i in range(num_sentences)
        ] if num_sentences > 0 else []

        # 4. Calculate Similarity Matrix & Avg Similarity
        # Veride 'similarities' sütunu olmadığı için embeddinglerden hesaplıyoruz.
        if len(node_embeddings) > 0:
            # Embedding listesini numpy array'e çevir
            embeddings_array = np.array(node_embeddings)

            # Boyut kontrolü (tek cümle varsa reshape gerekebilir)
            if embeddings_array.ndim == 1:
                embeddings_array = embeddings_array.reshape(1, -1)

            # Cosine similarity hesapla (N x N matris döner)
            similarities_matrix = cosine_similarity(embeddings_array)

            # Eşik değer için ortalamayı al
            avg_similarity = float(np.mean(similarities_matrix))
        else:
            similarities_matrix = np.empty((0, 0))
            avg_similarity = 0.0

        return {
            "nodes": node_embeddings,
            "sentences": sentences,
            "position_sentences": position_sentences,
            "similarities": similarities_matrix,
            "avg_similarity": avg_similarity,
        }

    def _create_graph_from_data(self, data: Dict[str, Any]) -> Data:
        """
        Creates the graph Data object from the extracted data dictionary.
        """
        nodes = data['nodes']
        similarities = data['similarities']
        avg_similarity = data['avg_similarity']
        position_sentences = data['position_sentences']

        # 1) Collect edge list based on similarity threshold
        edge_list: List[tuple[int, int]] = []
        edge_attr_list: List[list[float]] = []
        num_nodes = len(nodes)

        for i in range(num_nodes):
            for j in range(num_nodes):
                # Self-loop engelleme ve eşik değeri kontrolü
                if i != j and similarities[i, j] > avg_similarity:
                    edge_list.append((i, j))
                    # Edge features: [inverse distance, similarity score]
                    # distance = abs(i - j), yakın cümleler daha güçlü bağa sahip olabilir varsayımı
                    dist = abs(i - j)
                    inv_dist = 1.0 / dist if dist != 0 else 0.0  # Güvenlik önlemi

                    edge_attr_list.append(
                        [inv_dist, float(similarities[i, j])]
                    )

        # 2) Create edge_index and edge_attr tensors
        if edge_list:
            src, dst = zip(*edge_list)
            edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        else:
            # Handle graph with zero edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)

        # 3) Prepend position info to node features
        #    We create a copy to avoid modifying the input data list.
        node_features = [emb[:] for emb in nodes]  # Deep copy lists

        if node_features:
            for idx, emb in enumerate(node_features):
                # Feature olarak sadece pozisyon bilgisini ekliyoruz
                emb.insert(0, position_sentences[idx])

        # 4) Create node feature tensor
        x = torch.tensor(node_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def generate_from_row(
            self, row_data: Union[pd.DataFrame, pd.Series]
    ) -> Data:
        """
        Main entry point. Generates a graph from a single row of data.
        """
        extracted_data = self._extract_data_from_row(row_data)
        return self._create_graph_from_data(extracted_data)

    # Alias __call__ to the main generation method
    __call__ = generate_from_row

    def generate_batch(
            self,
            rows: Union[pd.DataFrame, Iterable[pd.Series]],
            max_workers: int = None
    ) -> Batch:
        """
        Generates a Batch object from multiple rows using parallel processing.
        """
        if isinstance(rows, pd.DataFrame):
            iterable = (rows.iloc[i] for i in range(len(rows)))
        else:
            iterable = rows

        def process_row(row_data):
            return GraphGenerator().generate_from_row(row_data)

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            data_list: List[Data] = list(
                exe.map(process_row, iterable)
            )

        return Batch.from_data_list(data_list)