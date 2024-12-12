def prompt_factory(strategy, row, ref_rows=None):
    if strategy == 1:
        return strategy_1(row, ref_rows)
    elif strategy == 2:
        return strategy_2(row, ref_rows)
    elif strategy == 3:
        prompt = (
            pre_intro_1()
            + info_1(row)
            + outro_1()
        )
        return prompt

def strategy_1(row, ref_rows=None):
    if not ref_rows is None: few_shot_prompt = "Here is some example information and required summary about the study:\n\n" + " ".join([info_1(ref_row) + outro_1() + get_summary(ref_row['summary']) for _, ref_row in ref_rows.iterrows()]) + "\nAccording to these examples write a lay summary for the following study:\n\n"
    else: few_shot_prompt = ""
    prompt = (
            pre_intro_1()
            + few_shot_prompt
            + info_1(row)
            + outro_1()
    )
    return prompt

def strategy_2(row, ref_rows=None):
    if not ref_rows is None: few_shot_prompt = "Here is some example information and required summary about the study:\n\n" + " ".join([info_1(ref_row) + outro_2() + get_summary(ref_row['summary']) for _, ref_row in ref_rows.iterrows()]) + "According to these examples write a lay summary for the following study:\n\n"
    else: few_shot_prompt = ""
    prompt = (
            pre_intro_2()
            + few_shot_prompt
            + info_1(row)
            + outro_2()
    )
    return prompt

def strategy_3(row):
    prompt = (
        "You are a scientific assistant specializing in creating concise and accurate lay summaries of biomedical research. "
        "Your task is to summarize the provided research paper into simple, clear language suitable for a general audience, "
        "while preserving the main findings and avoiding any additional interpretations or speculations. Focus only on the information provided.\n\n"
        "Here is the information about the study:\n\n"
        "1. **Title**:\n"
        f"{row['title']}\n\n"
        "2. **Abstract**:\n"
        f"{' '.join(map(str, row['abstract']))}\n\n"
        "3. **Selected Key Sentences from the Paper**:\n"
        f"{'\n'.join(map(str, row['rag_sentences']))}\n\n"
        "Using the information above, write a lay summary that meets the following criteria:\n"
        "- **Length**: Keep the summary between 4-6 sentences.\n"
        "- **Clarity**: Use simple and straightforward language suitable for non-experts.\n"
        "- **Focus**: Highlight the study's purpose, main findings, and implications without adding extra commentary or personal opinions.\n"
        "- **Avoid Technical Jargon**: Simplify complex terms wherever possible.\n\n"
        "Lay summary:"
    )
    return prompt

def pre_intro_1():
    prompt = (
        "You are a scientific assistant tasked with summarizing biomedical research papers. "
        "The goal is to create a simple and clear lay summary for a general audience.\n\n"

    )
    return prompt
def pre_intro_2():
    prompt = (
        "You are a scientific assistant specializing in creating concise and accurate lay summaries of biomedical research. "
        "Your task is to summarize the provided research paper into simple, clear language suitable for a general audience, "
        "while preserving the main findings and avoiding any additional interpretations or speculations. Focus only on the information provided.\n\n"
        "Using the information above, write a lay summary that meets the following criteria:\n"
        "- **Length**: Keep the summary between 4-6 sentences.\n"
        "- **Clarity**: Use simple and straightforward language suitable for non-experts.\n"
        "- **Focus**: Highlight the study's purpose, main findings, and implications without adding extra commentary or personal opinions.\n"
        "- **Avoid Technical Jargon**: Simplify complex terms wherever possible.\n\n"
    )
    return prompt

def info_1(row):
    prompt = (
        "Here is the title, abstract and selected sentence from sections of the paper:\n"
        "1. **Title**:\n"
        f"{row['title']}\n\n"
        "2. **Abstract**:\n"
        f"{' '.join(map(str, row['abstract']))}\n\n"
        "3. **Selected Key Sentences from the Paper**:\n"
        f"{'\n- '.join(map(str, row['rag_sentences']))}\n\n"
    )
    return prompt


def outro_1():
    prompt = (
        "Using the given information, write a lay summary in simple terms:\n"
    )
    return prompt
def outro_2():
    prompt = (
        "Lay summary:\n"
    )
    return prompt

def get_summary(summary):
    return " ".join(summary)
