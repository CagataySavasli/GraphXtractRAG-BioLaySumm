import torch
from torch.utils.data import Dataset
from lib.processors import GraphGenerator, SimilarityPreprocessor

class GraphDataloader(Dataset):
    def __init__(self, data):
        self.data = data
        self.graph_generator = GraphGenerator()
        self.similarity_preprocessor = SimilarityPreprocessor()


        self.data['sentences'] = self.data.apply(self.get_sentences, axis=1)
        self.data[['avg_similarity', 'similarities']] = self.data.apply(self.similarity_preprocessor, axis=1, result_type='expand')

    def get_sentences(self, row):
        sentences = [x for y in row['sections'] for x in y]
        return sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        graph_data = self.graph_generator(row)
        return row, graph_data