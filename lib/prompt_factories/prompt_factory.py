from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from lib.prompt_factories.prompt import ZeroShotPromptFactory, FewShotPromptFactory, LaySummaryPromptFactory
from lib.utility.case_builder import CaseBuilder
from typing import Union

class PromptFactory:
    def __init__(self):
        self.case_builder = CaseBuilder()
        self.strategy = self.case_builder.massage_strategy

    def build_factory(self, selected_sentences, ref_selected_sentencess=None):
        if self.strategy == "few_shot":
            return FewShotPromptFactory(selected_sentences, ref_selected_sentencess)
        elif self.strategy == "zero_shot":
            return ZeroShotPromptFactory(selected_sentences)
        elif self.strategy == "lay_summary":
            return LaySummaryPromptFactory(selected_sentences)
        else:
            raise ValueError(f"Unsupported prompt strategy: {self.strategy}")

    def _process_single(self, selected_sentences, ref_selected_sentencess):
        # Alt-fabrika oluştur ve prompt, instruction, info döndür
        factory = self.build_factory(selected_sentences, ref_selected_sentencess)
        prompt = factory.get_prompt()
        return prompt

    def get_prompt(self, selected_sentences, ref_selected_sentencess=None):
        return self._process_single(selected_sentences, ref_selected_sentencess)

    def get_batch(
        self,
        selected_sentences_batch: Union[pd.DataFrame, list],
        ref_selected_sentencess_batch: list = None,
        max_workers: int = None
    ) -> list:
        # 1) Satır listesini hazırlayın
        if isinstance(selected_sentences_batch, pd.DataFrame):
            selected_sentencess = [selected_sentences_batch.iloc[i] for i in range(len(selected_sentences_batch))]
        else:
            selected_sentencess = list(selected_sentences_batch)

        # 2) Referans satır listesini hazırla (few-shot için)
        if self.strategy == "few_shot":
            if ref_selected_sentencess_batch is None or len(ref_selected_sentencess_batch) != len(selected_sentencess):
                raise ValueError("ref_selected_sentencess_batch must be provided and match selected_sentences_batch length for few_shot strategy")
            refs = list(ref_selected_sentencess_batch)
        else:
            refs = [None] * len(selected_sentencess)

        # 3) Paralel işleme
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._process_single, selected_sentencess, refs))

        return results

    # __call__ ile hem tekil hem toplu kullanım
    __call__ = get_batch
