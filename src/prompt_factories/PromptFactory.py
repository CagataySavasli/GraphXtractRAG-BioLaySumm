from src.prompt_factories.FewShotPromptFactory import FewShotPromptFactory
from src.prompt_factories.ZeroShotPromptFactory import ZeroShotPromptFactory


class PromptFactory:
    def __init__(self, case_builder, row=None, ref_rows=None):
        strategy = case_builder.massage_strategy
        self.factory = self._get_factory(strategy, row, ref_rows)

    def _get_factory(self, strategy, row, ref_rows):
        if strategy == "few_shot":
            return FewShotPromptFactory(row, ref_rows)
        elif strategy == "zero_shot":
            return ZeroShotPromptFactory(row)
        else:
            raise ValueError("Unsupported strategy")

    def set_row(self, row, ref_rows=None):
        self.factory.set_row(row, ref_rows)

    def get_prompt(self):
        return self.factory.get_prompt()
