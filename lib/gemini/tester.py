import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

import re

class GeminiTester:
    def __init__(self):
        self.source_model = None
        self.genai_model = None

    def set_source_model(self, source_model):
        self.source_model = source_model

    def update_genai_model(self):
        self.genai_model = genai.GenerativeModel(self.source_model)

    def extract_summary(self, s: str) -> str:
        """
        Başta ve sonda harf veya rakam olana kadar
        boşluk ve noktalama işaretlerini temizler.
        """
        # ^[^A-Za-z0-9]+  -> baştan alfanümerik olmayanları
        # [^A-Za-z0-9]+$  -> sondan alfanümerik olmayanları

        if "lay_summary" in s:
            s = s.split("lay_summary")[1]

        s = re.sub(
            r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$',
            '',
            s
        )

        s = s + "."
        return s

    def predict(self, message: dict) -> tuple[str, str]:
        prompt = message["text_input"]
        label = message["output"]

        answer = self.genai_model.generate_content(prompt)
        # print(answer.text)
        # print("#############################################333")

        try:
            answer = answer.text
        except:
            answer = ""
        # print("Answer: \n", answer)
        clean_answer = self.extract_summary(answer) if answer != "" else answer

        return clean_answer, label

    def predict_batch(
            self,
            messages: list[dict],
            max_workers: int | None = None
    ) -> tuple[list[str], list[str]]:
        """
        Parallel batch prediction using ThreadPoolExecutor.

        Args:
            messages: List of dicts, each with "text_input" and "output" keys.
            max_workers: Number of threads to use (defaults to number of CPUs).

        Returns:
            List of (clean_answer, label) tuples in the same order as input messages.
        """
        results: list[tuple[str, str]] = [None] * len(messages)

        def _worker(idx: int, msg: dict):
            return idx, self.predict(msg)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, i, m): i for i, m in enumerate(messages)}
            for future in as_completed(futures):
                idx, res = future.result()
                results[idx] = res

        clean_answers, labels = zip(*results)
        return clean_answers, labels