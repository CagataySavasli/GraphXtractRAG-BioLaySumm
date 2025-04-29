from lib.prompt_factories.prompt_factory import PromptFactory

import pandas as pd

class MessageFactory:
    def __init__(self):

        self.prompt_generator = PromptFactory()

    def create_message(self, row: pd.Series, selected_sentences: list[list[str]]) -> list[ dict]:
        prompts = self.prompt_generator(selected_sentences)
        summaries = [str(summary) for summary in row['summary'].tolist()]

        message_list = []
        for prompt, summary in zip(prompts, summaries):
            message_list.append({"text_input": prompt, "output": summary})
        return message_list

    def create_message_row(self, row: pd.Series, selected_sentences: list[str]) -> dict:
        prompt = self.prompt_generator.get_prompt(selected_sentences)
        summary = str(row['summary'])

        return {"text_input": prompt, "output": summary}

    __call__ = create_message