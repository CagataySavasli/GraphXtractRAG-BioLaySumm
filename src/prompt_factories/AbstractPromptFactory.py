from abc import ABC, abstractmethod

class AbstractPromptFactory(ABC):
    def __init__(self, row=None, ref_rows=None):
        self.row = row
        self.ref_rows = ref_rows

    def set_row(self, row, ref_rows=None):
        self.row = row
        self.ref_rows = ref_rows

    @abstractmethod
    def get_prompt(self):
        pass

    @abstractmethod
    def get_instruction(self):
        pass

    @abstractmethod
    def few_info(self, row):
        pass

    @abstractmethod
    def info(self):
        pass
