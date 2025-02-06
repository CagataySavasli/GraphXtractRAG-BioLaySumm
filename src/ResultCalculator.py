from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import numpy as np
import evaluate
import textstat
import pandas as pd
from typing import Dict, List, Tuple


class ResultCalculator:
    def __init__(self, case_builder, results: Dict[str, List[str]] = None):
        """
        A class to calculate evaluation metrics.

        Computes ROUGE, BERTScore, BARTScore, FKGL, and DCRS for text summarization systems.
        """
        self.results = results
        self.batch_size = 5
        self.case_builder = case_builder

        # Evaluation metrics
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load("bertscore", device=case_builder.device)

        # Load BART model (for BARTScore computation)
        self.bart_model_name = "facebook/bart-large-cnn"
        self.bart_tokenizer = BartTokenizer.from_pretrained(self.bart_model_name)
        self.bart_model = BartForConditionalGeneration.from_pretrained(self.bart_model_name)

    def set_results(self, results: Dict[str, List[str]]) -> None:
        """
        Updates the results.
        """
        self.results = results

    def compute_bart_score(self, predictions: List[str], references: List[str]) -> List[float]:
        """
        Computes BARTScore.
        """
        bart_scores = []
        for pred, ref in zip(predictions, references):
            inputs = self.bart_tokenizer(ref, return_tensors="pt", truncation=True, max_length=1024)
            outputs = self.bart_tokenizer(pred, return_tensors="pt", truncation=True, max_length=1024)
            ref_to_pred_score = self.bart_model(**inputs, labels=outputs["input_ids"]).loss.item()
            pred_to_ref_score = self.bart_model(**outputs, labels=inputs["input_ids"]).loss.item()
            bart_scores.append((ref_to_pred_score + pred_to_ref_score) / 2)
        return bart_scores

    def get_rouge_results(self) -> Dict[str, float]:
        """
        Computes ROUGE metrics.
        """
        return self.rouge.compute(
            predictions=self.results['prediction'],
            references=self.results['reference'],
            use_aggregator=True,
            use_stemmer=True,
        )

    def get_bertscore_results(self) -> Dict[str, List[float]]:
        """
        Computes BERTScore metrics.
        """
        bertscore_results = {"precision": [], "recall": [], "f1": []}

        for idx in tqdm(range(0, len(self.results['prediction']), self.batch_size), desc="Calculating BERTScore"):
            batch_predictions = self.results['prediction'][idx: idx + self.batch_size]
            batch_references = self.results['reference'][idx: idx + self.batch_size]

            tmp_bertscore_results = self.bertscore.compute(
                predictions=batch_predictions,
                references=batch_references,
                model_type="microsoft/deberta-xlarge-mnli",
                device=self.case_builder.device,
            )

            bertscore_results["precision"].extend(tmp_bertscore_results["precision"])
            bertscore_results["recall"].extend(tmp_bertscore_results["recall"])
            bertscore_results["f1"].extend(tmp_bertscore_results["f1"])

        return bertscore_results

    def get_fkgl_dcrs_results(self) -> Tuple[List[float], List[float]]:
        """
        Computes FKGL and DCRS readability metrics.
        """
        fkgl_scores = [textstat.flesch_kincaid_grade(p) for p in self.results['prediction']]
        dcrs_scores = [textstat.dale_chall_readability_score(p) for p in self.results['prediction']]
        return fkgl_scores, dcrs_scores

    def get_bartscore_results(self) -> Dict[str, List[float]]:
        """
        Computes BARTScore metrics.
        """
        bart_scores = {"bart_scores": []}

        for idx in tqdm(range(0, len(self.results['prediction']), self.batch_size), desc="Calculating BARTScore"):
            batch_predictions = self.results['prediction'][idx: idx + self.batch_size]
            batch_references = self.results['reference'][idx: idx + self.batch_size]

            tmp_bart_scores = self.compute_bart_score(batch_predictions, batch_references)
            bart_scores["bart_scores"].extend(tmp_bart_scores)

        return bart_scores

    def get_results(self) -> pd.DataFrame:
        """
        Computes all metrics and returns a DataFrame.
        """
        rouge_results = self.get_rouge_results()
        bertscore_results = self.get_bertscore_results()
        fkgl_scores, dcrs_scores = self.get_fkgl_dcrs_results()
        bart_scores = self.get_bartscore_results()

        final_results = {
            "ROUGE1": [rouge_results['rouge1']],
            "ROUGE2": [rouge_results['rouge2']],
            "ROUGEL": [rouge_results['rougeL']],
            "BERTScore_Precision": [np.mean(bertscore_results["precision"])],
            "BERTScore_Recall": [np.mean(bertscore_results["recall"])],
            "BERTScore_F1": [np.mean(bertscore_results["f1"])],
            "FKGL": [np.mean(fkgl_scores)],
            "DCRS": [np.mean(dcrs_scores)],
            "BARTScore": [np.mean(bart_scores["bart_scores"])],
        }

        return pd.DataFrame(final_results)
