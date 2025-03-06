import torch


class SingletonMeta(type):
    """
    Singleton metaclass: Her sınıf için yalnızca bir örnek oluşturulmasını sağlar.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # Sınıf için ilk kez bir örnek oluşturuluyor
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class CaseBuilder(metaclass=SingletonMeta):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    def __init__(self, genai_version: str = "Gemini",
                 bert_version: str = "BiBERT",
                 massage_strategy: str = "zero_shot_performance_analyzer",
                 rag_strategy: str = "GESRAG",
                 rag_case: str = "top",
                 rag_n: int = 10,
                 dataset_name: str = "elife"):

        self.genai_version = genai_version
        self.bert_version = bert_version
        self.massage_strategy = massage_strategy
        self.rag_strategy = rag_strategy
        self.rag_case = rag_case
        self.rag_n = rag_n
        self.dataset_name = dataset_name

        if bert_version == 'BioBERT':
            self.bert_model_name = 'dmis-lab/biobert-base-cased-v1.2'

        if genai_version == 't5':
            self.genai_model_name = 'google/flan-t5-small'
        elif genai_version == 'BioMistral':
            self.genai_model_name = 'BioMistral/BioMistral-7B'
        elif genai_version == 'BioGBT':
            self.genai_model_name = 'microsoft/BioGPT-Large-PubMedQA'
        elif genai_version == 'Gemini':
            self.genai_model_name = 'models/gemini-1.5-flash-001-tuning'

    def get_case_signature(self):
        return f"{self.genai_version}_{self.bert_version}_{self.massage_strategy}_{self.rag_strategy}_{self.rag_case}_{self.dataset_name}.csv"
