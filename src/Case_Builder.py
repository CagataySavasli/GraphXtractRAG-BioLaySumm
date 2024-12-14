import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

genai_version = 'BioMistral'
massage_strategy = 'few_shot'
bert_version = 'BioBERT'
dataset_name = "elife"
prompt_strategy_used = 1



if bert_version == 'BioBERT':
    bert_model_name = 'dmis-lab/biobert-base-cased-v1.2'


if genai_version == 't5':
    genai_model_name = 'google/flan-t5-small'
elif genai_version == 'BioMistral':
    genai_model_name = 'BioMistral/BioMistral-7B'
elif genai_version == 'BioGBT':
    genai_model_name = 'microsoft/BioGPT-Large-PubMedQA'#'microsoft/biogpt'