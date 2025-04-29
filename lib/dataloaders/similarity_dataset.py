import torch
from torch.utils.data import Dataset

from lib.utility import CaseBuilder, MessageFactory
from lib.processors import SimilarityPreprocessor
from lib.rag_factories import RAG_Factory

class SimilarityDataset(Dataset):
    def __init__(self, data):
        self.data = data

        # Create required objects
        self.similarity_preprocessor = SimilarityPreprocessor()
        self.rag = RAG_Factory()
        self.message_factory = MessageFactory()


        self.data['sentences'] = self.data.apply(self.get_sentences, axis=1)
        self.selected_sentence = self.rag(self.data)
        self.messages = self.message_factory(self.data, self.selected_sentence)



    def get_sentences(self, row):
        sentences = [x for y in row['sections'] for x in y]
        return sentences

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = self.messages[idx]
        return message