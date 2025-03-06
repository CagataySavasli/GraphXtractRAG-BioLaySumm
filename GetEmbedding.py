from lib.utility.DatasetGenerator import DatasetGenerator
from lib.utility.CaseBuilder import CaseBuilder

import pandas as pd

# Initialize case builder and prompt factory
genai_type = "Gemini"
message_type = "zero_shot_performance_analyzer"
rag_type = "GESRAG"
rag_strategy = "bottom"
bert_model = 'BioBERT'

dataset = "elife"
dataset_info = "train"

case_builder = CaseBuilder(genai_type, bert_model, message_type, rag_type, rag_strategy, dataset)

# Load the dataset
df = pd.read_json(f'dataset/raw/elife/{dataset_info}.json')

dataset_generator = DatasetGenerator(case_builder)
dataset_generator.set_data(df)
dataset_generator.preprocess()
data = dataset_generator.get_data()

print(df.shape, data.shape)
data.to_json('src/dataset/processed/elife/train.json')