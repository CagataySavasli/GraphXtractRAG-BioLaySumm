import pandas as pd
import os

chunk = 100
chunk_list = []
data_path = "dataset/processed/plos/sep/"

for str_idx in range(0, 25000, chunk):
    end_idx = str_idx + chunk

    file_name = f"train_{str_idx}_{end_idx}.json"
    file_path = os.path.join(data_path, file_name)

    if os.path.isfile(file_path):
        tmp = pd.read_json(file_path)
        chunk_list.append(tmp)
    else:
        print(f"File {file_name} not found.")

data = pd.concat(chunk_list)
data.reset_index(drop=True, inplace=True)
print(data.shape)

data.to_json(f'dataset/processed/plos/train.json', orient='records')
