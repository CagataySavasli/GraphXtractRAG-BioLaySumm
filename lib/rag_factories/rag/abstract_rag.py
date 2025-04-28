from abc import ABC, abstractmethod


class AbstractRAG_Factory(ABC):

    @abstractmethod
    def get_n_sentences(self):
        pass
