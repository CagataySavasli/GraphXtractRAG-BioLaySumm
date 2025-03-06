from lib.prompt_factories.PromptFactory import PromptFactory
from lib.rag_factories.RAG_Factory import RAG_Factory

from lib.utility.CaseBuilder import CaseBuilder

import pandas as pd


class MessageFactory:
    def __init__(self):

        self.prompt_generator = PromptFactory()
        self.rag_calculator = RAG_Factory()

    def prepare_case(self, row: pd.Series):
        self.rag_calculator.set_row(row)
        row['rag_sentences'] = self.rag_calculator.get_n_sentences()
        self.prompt_generator.set_row(row)

    def get_instruction(self) -> str:
        instruction = self.prompt_generator.get_instruction()
        return instruction

    def get_prompt(self, row: pd.Series) -> str:
        self.prepare_case(row)
        prompt = self.prompt_generator.get_prompt()
        return prompt

    def get_info(self, row: pd.Series) -> str:
        self.prepare_case(row)
        info = self.prompt_generator.get_info()
        return info

    def create_message(self, row: pd.Series) -> dict:
        prompt = self.get_prompt(row)
        summary = str(row['summary'])

        return {"text_input": prompt, "output": summary}

    def create_message_prompting(self, row: pd.Series) -> dict:
        prompt = self.get_info(row)
        summary = str(row['summary'])

        return {"text_input": prompt, "output": summary}
