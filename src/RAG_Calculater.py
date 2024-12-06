import torch
import numpy as np
def iter_result_list(result_list, top_sentence, n):
    n_goal = len(result_list) + n

    for sentence in top_sentence:
        if not sentence in result_list:
            result_list.append(sentence)
            if len(result_list) == n_goal:
                break
    return result_list
def get_top_n_sentences(sentences, scores, n):
    combined = list(zip(scores, sentences))
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)

    top_n = sorted_combined[:n]
    top_n_sentences = [item[1] for item in top_n]

    return top_n_sentences


def get_top_n_articles(train_embedings, target_embeding, n):
    # Numpy array'lerini float32 formatına çevir
    train_embedings = np.array([np.array(embed, dtype=np.float32) for embed in train_embedings])
    target_embeding = np.array(target_embeding, dtype=np.float32)

    # PyTorch tensörüne dönüştür
    train_embedings = torch.tensor(train_embedings)
    target_embeding = torch.tensor(target_embeding)

    # Kosinüs benzerliğini hesapla
    similarities = torch.nn.functional.cosine_similarity(
        train_embedings.unsqueeze(1),  # Genişletme
        target_embeding.unsqueeze(0),  # Genişletme
        dim=2
    ).squeeze()

    _, top_n_indices = torch.topk(similarities, n, dim=0)

    return top_n_indices.tolist()
def RAG(row):
    n = 10
    result = {
        'sentence': [],
        'title': [],
        'abstract': [],
        'sum': []
    }
    row = [i for section in row for i in section]
    for item in row:
        result['sentence'].append(item['sentence'])
        result['title'].append(item['similarities'][0])
        result['abstract'].append(item['similarities'][1])
        result['sum'].append(sum(item['similarities']))

    top_n_title = get_top_n_sentences(result['sentence'], result['title'], n)
    top_n_abstract = get_top_n_sentences(result['sentence'], result['abstract'], n)
    top_n_sum = get_top_n_sentences(result['sentence'], result['sum'], n)

    rag_sentences = []
    rag_sentences = iter_result_list(rag_sentences, top_n_title, 2)
    rag_sentences = iter_result_list(rag_sentences, top_n_abstract, 3)
    rag_sentences = iter_result_list(rag_sentences, top_n_sum, 5)

    return rag_sentences