import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

dataset_name = "elife"
prompt_strategy_used = 1
bert_version = 'BioBERT'
genai_version = 't5'

if bert_version == 'BioBERT':
    bert_model_name = 'dmis-lab/biobert-base-cased-v1.2'

if genai_version == 't5':
    genai_model_name = 'google/flan-t5-small'