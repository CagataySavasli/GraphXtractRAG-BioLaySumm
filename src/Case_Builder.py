import torch


class CaseBuilder:
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    def __init__(self, genai_version, massage_strategy, bert_version, dataset_name, prompt_strategy_used):
        self.genai_version = genai_version
        self.massage_strategy = massage_strategy
        self.bert_version = bert_version
        self.dataset_name = dataset_name
        self.prompt_strategy_used = prompt_strategy_used

        if bert_version == 'BioBERT':
            self.bert_model_name = 'dmis-lab/biobert-base-cased-v1.2'

        if genai_version == 't5':
            self.genai_model_name = 'google/flan-t5-small'
        elif genai_version == 'BioMistral':
            self.genai_model_name = 'BioMistral/BioMistral-7B'
        elif genai_version == 'BioGBT':
            self.genai_model_name = 'microsoft/BioGPT-Large-PubMedQA'

    def get_case_signature(self):
        return f"{self.genai_version}_{self.massage_strategy}_{self.bert_version}_{self.dataset_name}_{self.prompt_strategy_used}.csv"
