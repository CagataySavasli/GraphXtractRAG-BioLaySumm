import pandas as pd
import sqlite3
import json
from torch.utils.data import Dataset
import ast

class DatabaseConnector(Dataset):
    def __init__(self, db_path='dataset/dataset.db', source_name='elife', split_name='train'):
        self.db_path = db_path
        self.source_name = source_name
        self.split_name = split_name

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM dataset WHERE source=? AND split=?",
                (self.source_name, self.split_name),
            )
            self.length = cursor.fetchone()[0]

    def set_split_name(self, split_name):
        self.split_name = split_name
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM dataset WHERE source=? AND split=?",
                (self.source_name, self.split_name),
            )
            self.length = cursor.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM dataset WHERE source=? AND split=? LIMIT 1 OFFSET ?",
                (self.source_name, self.split_name, idx),
            )
            row = cursor.fetchone()
            columns = [description[0] for description in cursor.description]
            row_dict = dict(zip(columns, row))

        # JSON string kolonları Python objelerine dönüştür
        for key, val in row_dict.items():
            try:
                row_dict[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass

        return pd.DataFrame([row_dict])

    def get_outer_pool(self) -> list[list[float]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT title_embedding FROM dataset WHERE source=? AND split=?",
                (self.source_name, self.split_name),
            )
            title_embeddings = cursor.fetchall()

        for idx, title_embedding in enumerate(title_embeddings):
            title_embedding = title_embedding[0]
            title_embeddings[idx] = ast.literal_eval(title_embedding)



        return title_embeddings
