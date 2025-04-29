import torch
from torch.utils.data import Dataset

from lib.utility import CaseBuilder, MessageFactory
from lib.processors import SimilarityPreprocessor, GraphGenerator
from lib.rag_factories import RAG_Factory

class PageRankDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.graph_generator = GraphGenerator()
        self.similarity_preprocessor = SimilarityPreprocessor()
        self.rag = RAG_Factory()
        self.message_factory = MessageFactory()


        self.data['sentences'] = self.data.apply(self.get_sentences, axis=1)
        self.data[['avg_similarity', 'similarities']] = self.data.apply(self.similarity_preprocessor, axis=1, result_type='expand')
        # self.graph_data = self.graph_generator.generate_batch(self.data)
        # self.selected_sentence = self.rag(self.data, self.graph_data)
        # self.messages = self.message_factory(self.data, self.selected_sentence)

    def get_sentences(self, row):
        sentences = [x for y in row['sections'] for x in y]
        return sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        graph_data = self.graph_generator(row)

        selected_sentence = self.rag.get_n_sentences(row, graph_data)
        message = self.message_factory.create_message_row(row, selected_sentence)
        return message