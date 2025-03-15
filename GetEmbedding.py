from lib.utility.DatasetGenerator import DatasetGenerator
from lib.utility.CaseBuilder import CaseBuilder
import sys
import pandas as pd

# Initialize case builder and prompt factory
genai_type = "Gemini"
message_type = "zero_shot_performance_analyzer"
rag_type = "GESRAG"
rag_strategy = "bottom"
bert_model = 'BioBERT'

dataset = sys.argv[1]
dataset_info = sys.argv[2]

case_builder = CaseBuilder(genai_type, bert_model, message_type, rag_type, rag_strategy, dataset)

# Load the dataset
df = pd.read_json(f'dataset/raw/{dataset}/{dataset_info}.json')

dataset_generator = DatasetGenerator()
dataset_generator.set_data(df)
dataset_generator.preprocess()
data = dataset_generator.get_data()

print(df.shape, data.shape)
data.to_json(f'dataset/processed/{dataset}/{dataset_info}.json')