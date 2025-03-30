from lib.utility.DatasetGenerator import DatasetGenerator
from lib.utility.CaseBuilder import CaseBuilder
import sys
import pandas as pd

dataset = sys.argv[1] # Used dataset name: ['elife', 'plos']
dataset_info = sys.argv[2] # Used dataset info: ['train', 'val', 'test']

# Lower and upper bound for the dataset to parallelize the process.
lower_bound = int(sys.argv[3])
upper_bound = int(sys.argv[4])


case_builder = CaseBuilder(dataset_name=dataset)

df = pd.read_json(f'dataset/raw/{dataset}/{dataset_info}.json')

df = df.loc[lower_bound:upper_bound].copy()

dataset_generator = DatasetGenerator()
dataset_generator.set_data(df)
dataset_generator.preprocess()
data = dataset_generator.get_data()

print(df.shape, data.shape)
data.to_json(f'dataset/processed/{dataset}/sep/{dataset_info}_{lower_bound}_{upper_bound}.json')