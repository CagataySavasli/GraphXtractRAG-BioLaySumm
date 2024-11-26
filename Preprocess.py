from src.Case_Builder import (device,
                              bert_version,
                              bert_model_name,
                              dataset_name
                              )
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import torch.nn.functional as F

print(device)

# Lead BERT Model
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertModel.from_pretrained(bert_model_name)
model.to(device)

data_train = pd.read_json(f'src/dataset/raw/{dataset_name}/train.json')
data_val = pd.read_json(f'src/dataset/raw/{dataset_name}/val.json')
data_test = pd.read_json(f'src/dataset/raw/{dataset_name}/test.json')

print(f"""
Dataset name: {dataset_name}
Train Data: {data_train.shape}
Validation Data: {data_val.shape}
Test Data: {data_test.shape}
""")


def cosine_similarity(tensor1, tensor2):
    """
    Compute the cosine similarity between two tensors.
    :param tensor1: PyTorch tensor of shape [1, 768]
    :param tensor2: PyTorch tensor of shape [1, 768]
    :return: Cosine similarity value (scalar)
    """
    # Normalize the tensors to have unit length
    tensor1_normalized = F.normalize(tensor1, p=2, dim=1)
    tensor2_normalized = F.normalize(tensor2, p=2, dim=1)

    # Compute cosine similarity as the dot product of the normalized vectors
    cosine_sim = torch.sum(tensor1_normalized * tensor2_normalized)

    return cosine_sim.item()


def get_similarity(row):
    """
    Verilen row içindeki `keywords` ve `sections` bilgilerini kullanarak
    cümlelerin keyword'lere göre benzerliklerini hesaplar.
    """
    global tmp, idx
    print(f"\r{idx}|{len(tmp)}|{(idx / len(tmp) * 100):.2f}", end=" ")

    idx += 1

    # Keyword tokenizasyonu
    rag_ref = []
    rag_ref.extend(row['title'])
    rag_ref.extend(" ".join(row['abstract']))
    rag_ref.extend(row['keywords'])
    keyword_tokens = tokenizer(rag_ref, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        keyword_embeddings = model(**keyword_tokens).pooler_output

    # Section ve cümle tokenizasyonu
    all_sentences = [sentence for section in row['sections'] for sentence in section]
    sentence_tokens = tokenizer(all_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        sentence_embeddings = model(**sentence_tokens).pooler_output

    # Kosinüs benzerliği hesaplama
    similarities = torch.nn.functional.cosine_similarity(
        sentence_embeddings.unsqueeze(1),
        keyword_embeddings.unsqueeze(0),
        dim=2)

    result_sections = []
    start_idx = 0
    for section in row['sections']:
        section_length = len(section)
        section_similarities = similarities[start_idx: start_idx + section_length]
        start_idx += section_length

        result_section = [
            {
                "sentence": sentence,
                "similarities": section_similarities[i].tolist()
            }
            for i, sentence in enumerate(section)
        ]
        result_sections.append(result_section)

    return result_sections


# data_train = data_train.iloc[:5].copy()
# data_val = data_val.iloc[:2].copy()
# data_test = data_test.iloc[:3].copy()


print("\nValidation : ")
tmp, idx = data_val.copy(), 1
data_val['sentences_similarity'] = data_val.apply(get_similarity, axis=1)
data_val.to_json(f'src/dataset/clean/{dataset_name}/{bert_version}_validation.json', orient='records')

print("\nTest : ")
tmp, idx = data_test.copy(), 1
data_test['sentences_similarity'] = data_test.apply(get_similarity, axis=1)
data_test.to_json(f'src/dataset/clean/{dataset_name}/{bert_version}_test.json', orient='records')

print("\nTrain : ")
tmp, idx = data_train.copy(), 1
data_train['sentences_similarity'] = data_train.apply(get_similarity, axis=1)
data_train.to_json(f'src/dataset/clean/{dataset_name}/{bert_version}_train.json', orient='records')