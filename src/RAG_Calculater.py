
def iter_result_list(result_list, top_sentence, n):
    n_goal = len(result_list) + n

    for sentence in top_sentence:
        if not sentence in result_list:
            result_list.append(sentence)
            if len(result_list) == n_goal:
                break
    return result_list
def get_top_n_sentences(sentences, scores, n):
    # Combine sentences and avg values into a single list
    combined = list(zip(scores, sentences))

    # Sort by avg values in descending order
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # Get top n sentences with highest avg values
    top_n = sorted_combined[:n]

    # Extract sentences and avg values separately
    top_n_sentences = [item[1] for item in top_n]

    return top_n_sentences


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