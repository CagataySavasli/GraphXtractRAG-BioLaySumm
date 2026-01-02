# %%
import pandas as pd
from tqdm import tqdm

from lib.model.model_chat import ModelChat
from lib.data.database_connector import DatabaseConnector
from lib.message.prompt_generator import PromptGenerator
# %%
# --- 1. SETTINGS and GENERATION ---
SELECTED_IDX = 0

TABLE_NAME = ["train", "validation"][1]

MODEL_PATH = ["llama3", "koesn/llama3-openbiollm-8b:q6_K", "gemma3", "alibayram/medgemma"][SELECTED_IDX]
MODEL_NAME = ["llama3", "biollama3", "gemma3", "medgemma"][SELECTED_IDX]

SAVED_FILE_NAME = f"{TABLE_NAME}_{MODEL_NAME}_zero_result.csv"

db_connector = DatabaseConnector(table_name=TABLE_NAME)
prompt_generator = PromptGenerator()
model_chat = ModelChat(MODEL_NAME, MODEL_PATH)
results = []

print(f"{SAVED_FILE_NAME} will be generated ... (Draft Results)")
# %%
i = 0
# --- Draft Result Generation Loop ---
for row in tqdm(db_connector, desc=f"Draft Results Generating with {MODEL_NAME}"):

    # --- STEP A: Get Data ---
    title = row['title'].values[0]
    abstract = " ".join(row['abstract'].values[0]) if isinstance(row['abstract'].values[0], list) else row['abstract'].values[0]
    original_summary = row['summary'].values[0] # Ground Truth

    # --- STEP B: Draft Step ---
    # 1. Generate Prompt
    draft_prompt_dict = prompt_generator.generate_draft_prompt(title, abstract)
    draft_instruction = draft_prompt_dict['instruction']
    draft_content = draft_prompt_dict['content']

    # 2. Ask Model (Generate Draft Summary)
    generated_draft_summary = ""
    count_draft = 0
    while len(generated_draft_summary) < 50:  # Minimum length check
        count_draft += 1
        if count_draft > 2:
            generated_draft_summary = draft_content # If answer is not getten, return content itself (fail-safe)
            break
        generated_draft_summary = model_chat.generate_zero_shot_answer(draft_content, system_instruction=draft_instruction)

    # --- STEP C: Save Answers ---
    results.append({
        "title": title,
        "abstract": abstract,
        "draft_prompt": draft_content,
        "generated_draft": generated_draft_summary,
        "ground_truth": " ".join(original_summary) if isinstance(original_summary, list) else original_summary,
    })

    i += 1
    if i == 10:
        break
# %%
df_result = pd.DataFrame(results)
df_result.to_csv(f"outputs/generated_texts/plos/draft/{SAVED_FILE_NAME}", index=False)
print(f"{SAVED_FILE_NAME} will be generated ... (Draft Results)")
# %%