from lib.rag_factories.rag.abstract_rag import AbstractRAG_Factory
from lib.utility.case_builder import CaseBuilder

import torch
import torch.nn.functional as F

class GraphXtractRAG(AbstractRAG_Factory):
    def __init__(self):
        # print("GraphXtractRAG Factory")

        self.case_builder = CaseBuilder()
        self.n = self.case_builder.rag_n

        self.model = None

        self.row = None
        self.sentences = None
        self.graph = None

    def set_row(self, row, graph):
        self.row = row
        self.sentences = list(row['sentences'])
        # print(len(self.sentences))
        self.graph = graph

    def select_sentences(self):
        with torch.no_grad():
            # logits = self.model(self.graph.x, self.graph.edge_index)  # shape: [num_nodes]
            #
            # n_sentence = min(self.n, len(self.graph.x))

            n_sentence = min(self.n, len(self.sentences))

            logits = self.graph

            # Logit'leri softmax ile olasılığa dönüştürüyoruz.
            probs = F.softmax(logits, dim=0)  # shape: [num_nodes]

            # Olasılık dağılımına göre en yüksek self.n değeri seçiliyor.
            topk_probs, node_indices = torch.topk(probs, n_sentence)

            # Seçilen düğümlerin log olasılıklarının toplamı hesaplanır.
            #selected_log_probs = torch.log(topk_probs).sum()

            # Seçilen düğümlere ait cümleleri alıyoruz.
            selected_sentences = [self.sentences[i] for i in node_indices.tolist()]

        return selected_sentences

    def get_n_sentences(self):
        return self.select_sentences()

