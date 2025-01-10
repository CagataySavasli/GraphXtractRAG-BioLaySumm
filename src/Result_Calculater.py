from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import numpy as np
import evaluate
import textstat
import pandas as pd


class ResultCalculater:
    def __init__(self, case_builder, results):
        # ROUGE and BERTScore
        self.results = results
        self.batch_size = 5
        self.case_builder = case_builder
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load("bertscore", device=case_builder.device)

        # Load a pre-trained BART model for BARTScore
        self.bart_model_name = "facebook/bart-large-cnn"
        self.bart_tokenizer = BartTokenizer.from_pretrained(self.bart_model_name)
        self.bart_model = BartForConditionalGeneration.from_pretrained(self.bart_model_name)

    # Compute BARTScore for Factuality
    def compute_bart_score(self, predictions, references):
        bart_scores = []
        for pred, ref in zip(predictions, references):
            inputs = self.bart_tokenizer(ref, return_tensors="pt", truncation=True, max_length=1024)
            outputs = self.bart_tokenizer(pred, return_tensors="pt", truncation=True, max_length=1024)
            ref_to_pred_score = self.bart_model(**inputs, labels=outputs["input_ids"]).loss.item()
            pred_to_ref_score = self.bart_model(**outputs, labels=inputs["input_ids"]).loss.item()
            bart_scores.append((ref_to_pred_score + pred_to_ref_score) / 2)
        return bart_scores

    def get_rouge_results(self):
        # Compute ROUGE metrics
        # print("ROUGE Metrics Calculater:")
        rouge_results = self.rouge.compute(
            predictions=self.results['prediction'],
            references=self.results['reference'],
            use_aggregator=True,
            use_stemmer=True,
        )
        return rouge_results

    def get_bertscore_results(self):
        # Compute BERTScore
        # print("BERTScore Calculater:")
        bertscore_results = {
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for idx in tqdm(range(0, len(self.results), self.batch_size)):
            str_idx = idx
            end_idx = idx + self.batch_size
            tmp_bertscore_results = self.bertscore.compute(
                predictions=self.results['prediction'][str_idx:end_idx].to_list(),
                references=self.results['reference'][str_idx:end_idx].to_list(),
                model_type="microsoft/deberta-xlarge-mnli",
                device=self.case_builder.device,
            )
            bertscore_results["precision"].extend(tmp_bertscore_results["precision"])
            bertscore_results["recall"].extend(tmp_bertscore_results["recall"])
            bertscore_results["f1"].extend(tmp_bertscore_results["f1"])

        return bertscore_results

    def get_fkgl_dcrs_results(self):
        # Compute FKGL and DCRS for Readability
        # print("FKGL Metrics Calculater:")
        fkgl_scores = [textstat.flesch_kincaid_grade(p) for p in self.results['prediction'].to_list()]
        # print("DCRS Metrics Calculater:")
        dcrs_scores = [textstat.dale_chall_readability_score(p) for p in self.results['prediction'].to_list()]
        return fkgl_scores, dcrs_scores

    def get_bartscore_results(self):
        # Compute BARTScore
        # print("BARTScore Calculater:")
        bart_scores = {
            "bart_scores": [],
        }
        for idx in tqdm(range(0, len(self.results), self.batch_size)):
            str_idx = idx
            end_idx = idx + self.batch_size
            tmp_bart_scores = self.compute_bart_score(self.results['prediction'][str_idx:end_idx].to_list(),
                                                      self.results['reference'][str_idx:end_idx].to_list())
            bart_scores["bart_scores"].extend(tmp_bart_scores)
        return bart_scores
    def get_results(self):

        rouge_results = self.get_rouge_results()
        bertscore_results = self.get_bertscore_results()
        fkgl_scores, dcrs_scores = self.get_fkgl_dcrs_results()
        bart_scores = self.get_bartscore_results()

        final_results = {
            "ROUGE1": [rouge_results['rouge1']],
            "ROUGE2": [rouge_results['rouge2']],
            "ROUGEL": [rouge_results['rougeL']],
            "BERTScore_Precision": [np.average(bertscore_results["precision"])],
            "BERTScore_Recall": [np.average(bertscore_results["recall"])],
            "BERTScore_F1": [np.average(bertscore_results["f1"])],
            "FKGL": [np.average(fkgl_scores)],
            "DCRS": [np.average(dcrs_scores)],
            "BARTScore": [np.average(bart_scores["bart_scores"])],
        }

        result_df = pd.DataFrame(final_results)

        return result_df
