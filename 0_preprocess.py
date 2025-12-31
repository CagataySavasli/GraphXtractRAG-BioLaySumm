
from lib import DataLoaderReloader, Preprocessor
import pandas as pd
from tqdm import tqdm

# 1. tqdm'in pandas entegrasyonunu başlatın
tqdm.pandas()

DATA_NAME = ["plos", "elife"][0]
SPLIT = ["train", "validation", "test"][1]

local_save_path = f"dataset/processed/{DATA_NAME}/sep/"
if SPLIT == "validation": SPLIT_ = "val"
else: SPLIT_ = SPLIT
referance_dataset_path = f"dataset/raw/{DATA_NAME}/{SPLIT_}.json"

data_loader = DataLoaderReloader(DATA_NAME)
preprocessor = Preprocessor()

referance_dataset = data_loader.reload()
referance_dataset = referance_dataset[SPLIT]

dataset = pd.read_json(referance_dataset_path)

referance_titles = referance_dataset['title']
mask = dataset['title'].isin(referance_titles)

df_all = dataset[mask]

print(f"""
Control Datasets:

RAW DATA: {len(referance_dataset)}
PREPROCESSED DATA: {len(df_all)}
""")

for str_idx in range(0, len(df_all), 500):
    end_idx = min(str_idx + 500, len(df_all))
    save_path = f"{local_save_path}{SPLIT}_{str_idx}_{end_idx}.parquet"
    print(f"Processing rows {str_idx} to {end_idx - 1}...")
    df_batch = df_all[str_idx:end_idx].copy()

    df_batch['title_embeddings'] = df_batch['title'].progress_apply(preprocessor.get_embedding_one_sentence)

    df_batch['abstract_embeddings'] = df_batch['abstract'].progress_apply(preprocessor.get_embedding_one_section)

    df_batch['section_embeddings'] = df_batch['sections'].progress_apply(preprocessor.get_embedding_multi_sections)

    df_batch.to_parquet(
            save_path,
            engine='pyarrow',  # 'pyarrow' veya 'fastparquet'
            index=False
        )
    print(f"Dataset saved to {save_path}")


