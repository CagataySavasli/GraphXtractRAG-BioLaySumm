from src.Prompt_Factory import prompt_factory
from src.Case_Builder import prompt_strategy_used


def massage_factory(strategy, text, ref_row=None):
    if strategy == 'zero_shot':
        return zero_shot(text)
    elif strategy == 'few_shot':
        return few_shot(text, ref_row)


def zero_shot(row):
    prompt = prompt_factory(prompt_strategy_used, row)
    massege = [{"role": "user", "content": prompt}]
    return massege


def few_shot(target_row, ref_row):
    massege = []
    for i in range(len(ref_row)):
        ref_prompt = prompt_factory(ref_row.loc[i])
        massege.append({"role": "user", "content": ref_prompt})
        massege.append({"role": "assistant", "content": ref_row.loc[i, 'summary']})

    prompt = prompt_factory(prompt_strategy_used, target_row)
    massege.append({"role": "user", "content": prompt})
    return massege
