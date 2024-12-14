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
    few_shot_prompt = ""
    if not ref_rows is None:
        few_shot_prompt += "# EXAMPLES:\n\n"
        for idx , ref_row in ref_rows.iterrows():
            few_shot_prompt += f"## Example {idx+1}:\n"
            few_shot_prompt += few_info_2(ref_row)
            few_shot_prompt += "---\n\n"


    prompt = (
            pre_intro_2()
            + few_shot_prompt
            + "For the given inputs, first generate your reasoning and then generate the outputs.\n\n"
            + info_2(row)
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

def pre_intro_2():
    prompt = (
        "# TASK:\n"
        "Craft a succinct and straightforward lay summary aimed at an audience without a specialized background in the subject. Leverage the title, abstract, and key sentences provided to ensure your summary encapsulates the essence and findings of the scientific research. Here is a structured breakdown to guide your summary:\n\n"
    
        "1. **Title Information**: Extract the main topic or issue addressed in the study from the title.\n"
        "2. **Abstract Details**: From the abstract, distill the primary purpose and approach of the research.\n"
        "3. **Key Findings from Selected Sentences**: Integrate the core discoveries or conclusions highlighted in the selected key sentences.\n\n"
        
        "Your goal is to concisely relay the scientific insights in terms that a layperson can understand, ensuring the summary is educational yet engaging. Maintain a length of 4-6 sentences, and focus strictly on the material provided without inferring additional data or voicing personal interpretations. Strive for clarity by avoiding or simplifying scientific jargon to keep the text accessible and relatable to the general public.\n"
        "---\n"
        
        "# FORMAT:\n"
        "Follow the following format:\n\n"
        
        "## INPUT:\n"
        "title: Title of the research paper\n"
        "abstract: Abstract of the research paper, providing a brief summary\n"
        "selected_key_sentences: Key sentences selected from different sections of the paper\n"
        "## OUTPUT:\n"
        "lay_summary: A concise, clear summary of the research paper suitable for a general audience, adhering to specified criteria (length, clarity, focus, technical jargon avoidance)\n\n"

        "---\n"
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

def few_info_2(row):
    prompt = (
        "## INPUT:\n"
        f"title: {row['title']}\n"
        f"abstract: [{' '.join(map(str, row['abstract']))}]\n"
        f"selected_key_sentences: {str(row['rag_sentences'])}\n"
        "## OUTPUT:\n"
        f"lay_summary: {' '.join(map(str, row['summary']))}\n\n"

    )
    return prompt
def info_2(row):
    prompt = (
        "## INPUT:\n"
        f"title: {row['title']}\n"
        f"abstract: [{' '.join(map(str, row['abstract']))}]\n"
        f"selected_key_sentences: {str(row['rag_sentences'])}\n"
        "## OUTPUT:\n"
        f"lay_summary: \n\n"

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
