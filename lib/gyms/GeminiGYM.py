from lib.utility.MessageFactory import MessageFactory
from lib.utility.ResultCalculator import ResultCalculator
from lib.utility.CaseBuilder import CaseBuilder

import google.generativeai as genai

from tqdm import tqdm
import pandas as pd
import time

genai.configure(api_key="AIzaSyC42OyqZc03g56rzaoC4JkDV9dt7TZ49ic")  # Write Your Gemini API Key


class GeminiGYM:
    def __init__(self):
        self.case_builder = CaseBuilder()
        self.n = self.case_builder.rag_n

        self.result_calculater = ResultCalculator()

        self.train_data = None
        self.test_data = None

        self.base_model = self.case_builder.genai_model_name
        self.genai_model = None

        self.y_true = None
        self.y_pred = None

        self.result = None

    def set_test_data(self, test_data):
        self.test_data = test_data.copy()

    def set_train_data(self, train_data):
        self.train_data = train_data.copy()

    def set_base_model(self, base_model):
        self.base_model = base_model

    def get_messages(self):
        massage_factory = MessageFactory()

        train_messages = []
        for _, row in self.train_data.iterrows():
            message = massage_factory.create_message(row)
            train_messages.append(message)
        return train_messages

    def fine_tune(self, display_name, epoch_count):
        train_messages = self.get_messages()

        operation = genai.create_tuned_model(
            # You can use a tuned model here too. Set `source_model="tunedModels/..."`
            display_name=display_name,
            source_model=self.base_model,
            epoch_count=epoch_count,
            batch_size=self.case_builder.batch_size,
            learning_rate=self.case_builder.lr,
            training_data=train_messages
        )

        for status in operation.wait_bar():
            time.sleep(5)

        self.result = operation.result()

        self.genai_model = genai.GenerativeModel(self.result.name)

    def evaluate(self):
        massage_factory = MessageFactory()

        test_results = {
            "true": [],
            "pred": []
        }
        for _, row in tqdm(self.test_data.iterrows(), total=len(self.test_data), desc="Test Process"):
            message = massage_factory.create_message(row)

            prompt = message["text_input"]
            label = message["output"]

            answer = self.generate_summary(prompt)

            test_results["true"].append(label)
            test_results["pred"].append(answer)

        return test_results

    def update_model(self):
        self.base_model = self.result.name

    def generate_summary(self, prompt_text: str, max_retries: int = 3, retry_delay: float = 5.0) -> str:
        """
        Generates a lay summary using the Gemini AI model.
        Includes automatic retries in case of API failures.

        Args:
            row (pd.Series): A row of dataframe containing the data needed for prompt generation.
            max_retries (int): Maximum number of retries for handling API errors.
            retry_delay (float): Time (in seconds) to wait before retrying after a failure.

        Returns:
            Optional[str]: The generated summary if successful, None if the request fails.
        """
        for attempt in range(1, max_retries + 1):
            try:
                # Request AI model to generate content
                response = self.genai_model.generate_content(prompt_text)

                # Extract and return summary from response
                return self.extract_summary(response.text)

            except Exception as error:
                print(f"Attempt {attempt} failed: {error}")

                # If it's the last attempt, return None
                if attempt == max_retries:
                    print("Maximum retries reached. Returning None.")
                    return ""

                # if attempt % 2 == 0:
                #     # Initialize the Generative AI model
                #     self.reset_generaoi_model()

                # Wait before retrying the request
                time.sleep(retry_delay)

    def extract_summary(self, response_text: str) -> str:
        """
        Extracts the summary from the AI model response.

        Args:
            response_text (str): The raw response text from the AI model.

        Returns:
            str: Extracted summary text.
        """
        # Check for different possible summary formats in the response
        if "lay_summary:" in response_text:
            return response_text.split("lay_summary:")[1].strip()
        elif "**Lay Summary:**" in response_text:
            return response_text.split("**Lay Summary:**")[1].strip()
        else:
            return response_text.strip()
