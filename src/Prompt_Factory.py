def prompt_factory(strategy, row):
    if strategy == 1:
        return strategy_1(row)
    elif strategy == 2:
        return strategy_2(row)

def strategy_1(row):
    prompt = (
        "You are a scientific assistant tasked with summarizing biomedical research papers. "
        "The goal is to create a simple and clear lay summary for a general audience. "
        "Here is the title and abstract of the paper:\n\n"
        "Title:\n"
        f"{row['title']}\n"
        "Abstract:\n"
        f"{' '.join(map(str, row['abstract']))}\n"  # Ensure all elements in 'abstract' are strings
        "\n\nUsing the given information, write a lay summary in simple terms:"
    )
    return prompt

def strategy_2(row):
    prompt = (
        "You are a scientific assistant tasked with summarizing biomedical research papers. "
        "The goal is to create a simple and clear lay summary for a general audience. "
        "Here is the title, abstract and selected sentence from sections of the paper:\n\n"
        "Title:\n"
        f"{row['title']}\n"
        "Abstract:\n"
        f"{' '.join(map(str, row['abstract']))}\n"  # Ensure all elements in 'abstract' are strings
        "Selected sentence:\n"
        f"{'\n'.join(map(str, row['rag_sentences']))}\n"  # Ensure all elements in 'rag_sentences' are strings
        "\n\nUsing the given information, write a lay summary in simple terms:"
    )
    return prompt