from lib.rag_factories.AbstractRAG_Factory import AbstractRAG_Factory
from lib.selectors.MIXSelector import MIXSelector
from lib.utility.GraphGenerator import GraphGenerator
import torch
import torch.nn.functional as F
class GESRAG(AbstractRAG_Factory):
    def __init__(self, n):
        self.model = MIXSelector(770, 128)
        self.model.load_state_dict(torch.load("./outputs/models/MIX_20_selector.pth", weights_only=True))
        self.model.eval()

        self.n =n

        self.graph_generator = GraphGenerator()

        self.row = None
        self.sentences = None

    def set_row(self, row):
        self.row = row
        self.graph_generator.set_row(row)
        self.sentences = self.graph_generator.get_sentences()

    def select_sentences(self, data):

        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)  # shape: [num_nodes]

            # Adım 2: Logit'leri softmax ile olasılığa dönüştürüyoruz.
            probs = F.softmax(logits, dim=0)  # shape: [num_nodes]

            # Adım 3: Olasılık dağılımından n düğümü örnekliyoruz.
            node_indices = torch.multinomial(probs, self.n, replacement=False)

            # Adım 4: Seçilen düğümlerin log olasılıklarının toplamı hesaplanır.
            selected_log_probs = torch.log(probs[node_indices]).sum()

            # Adım 5: Seçilen düğümlere ait cümleleri birleştirip prompt oluşturuyoruz.
            selected_sentences = [self.sentences[i] for i in node_indices.tolist()]

        return selected_sentences

    def get_n_sentences(self):
        data = self.graph_generator.create_graph()
        return self.select_sentences(data)