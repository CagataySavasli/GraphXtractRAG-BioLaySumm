from abc import ABC, abstractmethod

class AbstractPromptFactory(ABC):
    def __init__(self, selected_sentences=None, ref_selected_sentencess=None):
        self.selected_sentences = selected_sentences
        self.ref_selected_sentencess = ref_selected_sentencess

    def set_selected_sentences(self, selected_sentences, ref_selected_sentencess=None):
        self.selected_sentences = selected_sentences
        self.ref_selected_sentencess = ref_selected_sentencess

    @abstractmethod
    def get_prompt(self):
        pass

    @abstractmethod
    def get_instruction(self):
        pass

    @abstractmethod
    def few_info(self, selected_sentences):
        pass

    @abstractmethod
    def info(self):
        pass
