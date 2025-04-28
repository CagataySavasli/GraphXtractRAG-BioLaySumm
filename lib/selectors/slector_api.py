from models import GATGCNSelector
from lib.utility.case_builder import CaseBuilder

import torch

class SelectorAPI:
    def __init__(self):

        self.model = GATGCNSelector()

        self.case_builder = CaseBuilder()
        self.n = self.case_builder.rag_n


    def predict(self, graph):
        with torch.no_grad():
            logits = self.model(graph.x, graph.edge_index)  # shape: [num_nodes]

        return logits