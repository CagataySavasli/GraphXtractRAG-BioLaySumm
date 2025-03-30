import os, sys, json
import textstat
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score
import nltk
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

nltk.download('punkt')


class ResultCalculator:

    def __init__(self):
        self.biolaysumm_24 = pd.read_csv('./dataset/biolaysumm_24.csv')

        self.lambdas = {
            'ROUGE1': 0.2,
            'ROUGE2': 0.1,
            'ROUGEL': 0.4,
            'BERTScore': 0.3,
            'FKGL': -0.2,
            'DCRS': -0.2,
            'CLI': 0.0,
            #'BARTScore': 0.3
        }

    def calc_rouge(self, preds, refs):
        # Get ROUGE F1 scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], \
                                          use_stemmer=True, split_summaries=True)
        scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
        return np.mean([s['rouge1'].fmeasure for s in scores]), \
            np.mean([s['rouge2'].fmeasure for s in scores]), \
            np.mean([s['rougeLsum'].fmeasure for s in scores])

    def calc_bartscore(self, preds, srcs):
        # Get BARTScore scores
        bart_scorer = BARTScorer(device='cpu', max_length=8192)
        return np.mean(bart_scorer.score(srcs, preds))
    def calc_bertscore(self, preds, refs):
        # Get BERTScore F1 scores
        P, R, F1 = score(preds, refs, lang="en", verbose=False)
        return np.mean(F1.tolist())


    def calc_readability(self, preds):
        fkgl_scores = []
        cli_scores = []
        dcrs_scores = []
        for pred in preds:
            fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
            cli_scores.append(textstat.coleman_liau_index(pred))
            dcrs_scores.append(textstat.dale_chall_readability_score(pred))
        return np.mean(fkgl_scores), np.mean(cli_scores), np.mean(dcrs_scores)


    def evaluate(self, preds, refs):

        score_dict = {}

        # Relevance scores
        rouge1_score, rouge2_score, rougel_score = self.calc_rouge(preds, refs)
        score_dict['ROUGE1'] = rouge1_score
        score_dict['ROUGE2'] = rouge2_score
        score_dict['ROUGEL'] = rougel_score
        score_dict['BERTScore'] = self.calc_bertscore(preds, refs)

        # Readability scores
        fkgl_score, cli_score, dcrs_score = self.calc_readability(preds)
        score_dict['FKGL'] = fkgl_score
        score_dict['DCRS'] = dcrs_score
        score_dict['CLI'] = cli_score

        # Factuality scores
        #score_dict['BARTScore'] = self.calc_bartscore(preds, refs)#self.bart_factuality_evaluator.score(preds, refs)

        return score_dict

    @staticmethod
    def normalize_metric(value, min_val, max_val):
        """Verilen değeri belirtilen aralıkta 0-1 ölçeğine çeker."""
        value = max(min_val, min(value, max_val))
        return (value - min_val) / (max_val - min_val)

    def reward_function(self, pred, ref):
        if type(pred) == str:
            pred = [pred]
            ref = [ref]

        scores = self.evaluate(pred, ref)

        # Normalize scores
        scores['FKGL'] = self.normalize_metric(scores['FKGL'], 0, 20)
        scores['DCRS'] = self.normalize_metric(scores['DCRS'], 0, 12)

        reward = sum([scores[key] * self.lambdas[key] for key in scores.keys()])

        return reward

    def calculate_rank(self, metric, scores):
        pre_scores = self.biolaysumm_24[metric].tolist()

        if metric in ['ROUGE1', 'ROUGE2', 'ROUGEL', 'BERTScore']:
            fixed_value = round(scores * 100, 2)
            pre_scores.append(fixed_value)
            pre_scores.sort(reverse=True)
        else:
            fixed_value = round(scores, 2)
            pre_scores.append(fixed_value)
            pre_scores.sort(reverse=False)
        rank = pre_scores.index(fixed_value) + 1
        return rank, fixed_value

    def display_rank(self, scores):
        for key, value in scores.items():
            rank, fixed_value = self.calculate_rank(key, value)
            print(f'{key}: {fixed_value} | Rank: {rank}')

# %%
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
from transformers import LEDTokenizer
from transformers import LEDForConditionalGeneration

class BARTScorer:
    def __init__(self, device='cpu', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        # self.tokenizer = LEDTokenizer.from_pretrained(checkpoint)
        # self.model = LEDForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=2):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))


