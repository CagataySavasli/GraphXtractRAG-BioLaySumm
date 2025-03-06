from lib.prompt_factories.FewShotPromptFactory import FewShotPromptFactory
from lib.prompt_factories.ZeroShotPromptFactory import ZeroShotPromptFactory
from lib.prompt_factories.ZeroShotPerformanceAnalyzerPromptFactory import ZeroShotPerformanceAnalyzerFactory
from lib.utility.CaseBuilder import CaseBuilder

class PromptFactory:
    def __init__(self, row=None, ref_rows=None):
        self.case_builder = CaseBuilder()
        strategy = self.case_builder.massage_strategy
        self.factory = self.build_factory(strategy, row, ref_rows)

    def build_factory(self, strategy, row, ref_rows):
        if strategy == "few_shot":
            return FewShotPromptFactory(row, ref_rows)
        elif strategy == "zero_shot":
            return ZeroShotPromptFactory(row)
        elif strategy == "zero_shot_performance_analyzer":
            return ZeroShotPerformanceAnalyzerFactory(row)
        else:
            raise ValueError("Unsupported prompt strategy")

    def set_row(self, row, ref_rows=None):
        self.factory.set_row(row, ref_rows)

    def get_prompt(self):
        return self.factory.get_prompt()

    def get_instruction(self):
        return self.factory.get_instruction()

    def get_info(self):
        return self.factory.info()
