from src.prompt_factories.PromptFactory import PromptFactory
from src.rag_factories.RAG_Factory import RAG_Factory


class MassegeFactory:
    def __init__(self, case_builder, n, n_2=None, n_3=None):
        self.case_builder = case_builder
        self.n = n
        self.n_2 = n_2
        self.n_3 = n_3

        self.prompt_generator = PromptFactory(self.case_builder)
        self.rag_calculator = RAG_Factory(self.case_builder, self.n, self.n_2, self.n_3)

    def prepare_case(self, row):
        self.rag_calculator.set_row(row)
        row['rag_sentences'] = self.rag_calculator.get_n_sentences()
        self.prompt_generator.set_row(row)

    def get_instruction(self):
        instruction = self.prompt_generator.get_instruction()
        return instruction

    def get_prompt(self, row):
        self.prepare_case(row)
        prompt = self.prompt_generator.get_prompt()
        return prompt

    def get_info(self, row):
        self.prepare_case(row)
        info = self.prompt_generator.get_info()
        return info


    def create_massege(self, row):
        prompt = self.get_prompt(row)
        summary = str(row['summary'])

        return {"text_input": prompt, "output": summary}

    def create_massege_prompting(self, row):
        prompt = self.get_info(row)
        summary = str(row['summary'])

        return {"text_input": prompt, "output": summary}
