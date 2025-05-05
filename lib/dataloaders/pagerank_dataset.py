import torch
from torch.utils.data import Dataset
import sqlite3
import json
import pandas as pd

from lib.utility import CaseBuilder, MessageFactory
from lib.processors import SimilarityPreprocessor, GraphGenerator
from lib.rag_factories import RAG_Factory

class PageRankDataset(Dataset):
    def __init__(self, db_path='dataset/dataset.db', source_name='elife', split_name='train'):
        self.db_path = db_path
        self.source = source_name
        self.split = split_name

        self.graph_generator = GraphGenerator()
        self.similarity_preprocessor = SimilarityPreprocessor()
        self.rag = RAG_Factory()
        self.message_factory = MessageFactory()

        # Connect to SQLite DB to get length
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dataset WHERE source=? AND split=?",
                       (self.source, self.split))
        self.length = cursor.fetchone()[0]
        conn.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM dataset WHERE source=? AND split=? LIMIT 1 OFFSET ?",
                       (self.source, self.split, idx))
        row = cursor.fetchone()
        columns = [description[0] for description in cursor.description]
        row_dict = dict(zip(columns, row))

        conn.close()

        # Convert JSON string columns back to Python objects
        for key, val in row_dict.items():
            try:
                row_dict[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass

        # Convert to DataFrame for compatibility with processors
        row_df = pd.DataFrame([row_dict])

        # Apply processing
        row_dict['sentences'] = self.get_sentences(row_dict)
        avg_similarity, similarities = self.similarity_preprocessor(row_dict)
        row_dict['avg_similarity'] = avg_similarity
        row_dict['similarities'] = similarities

        graph_data = self.graph_generator(row_dict)
        selected_sentence = self.rag.get_n_sentences(row_dict, graph_data)
        message = self.message_factory.create_message_row(row_dict, selected_sentence)

        return message

    def get_sentences(self, row):
        sentences = [x for y in row['sections'] for x in y]
        return sentences
