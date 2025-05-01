from lib.utility import CaseBuilder

import google.generativeai as genai
import time

class GeminiFineTuner:
    def __init__(self):
        self.case_builder = CaseBuilder()

        display_name = f"{self.case_builder.rag_strategy}_{self.case_builder.rag_n}"
        self.display_name = display_name

        self.batch_size = self.case_builder.batch_size
        self.learning_rate = self.case_builder.lr
        self.source_model = self.case_builder.genai_model_name

        self.epoch_count = None
        self.training_data = None

        self.result = None


    def set_epoch_count(self, epoch_count: int):
        self.epoch_count = epoch_count

    def set_training_data(self, training_data: list[dict]):
        self.training_data = training_data

    def set_model(self, source_model: str):
        self.source_model = source_model

    def fit(self):
        operation = genai.create_tuned_model(
            # You can use a tuned model here too. Set `source_model="tunedModels/..."`
            display_name=self.display_name,
            source_model=self.source_model,
            epoch_count=self.epoch_count,
            batch_size=self.case_builder.batch_size,
            learning_rate=self.case_builder.lr,
            training_data=self.training_data
        )

        for status in operation.wait_bar():
            time.sleep(5)

        self.result = operation.result()

    def get_fine_tuned_model_name(self):
        return self.result.name

