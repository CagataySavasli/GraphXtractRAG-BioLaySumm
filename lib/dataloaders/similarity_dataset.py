import torch
from torch.utils.data import Dataset
import pandas as pd
import sqlite3
import json

from lib.utility import CaseBuilder, MessageFactory
from lib.processors import SimilarityPreprocessor
from lib.rag_factories import RAG_Factory

class SimilarityDataset(Dataset):
    def __init__(self, db_path='dataset/dataset.db', source_name='elife', split_name='train'):
        self.db_path = db_path
        self.source = source_name
        self.split = split_name

        # Initialize processors and factories
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

        # Retrieve the specific row by index
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

        # Process sentences using your previous logic
        row_dict['sentences'] = self.get_sentences(row_dict)

        selected_sentence = self.rag(pd.DataFrame([row_dict]))
        message = self.message_factory(pd.DataFrame([row_dict]), selected_sentence)[0]

        return message

    def get_sentences(self, row):
        sentences = [x for y in row['sections'] for x in y]
        return sentences
