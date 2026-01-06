# %%
import sys
import pandas as pd
from tqdm import tqdm

from lib.model.model_chat import ModelChat
from lib.data.database_connector import DatabaseConnector
from lib.message.prompt_generator import PromptGenerator
# %%
# --- 1. SETTINGS and GENERATION ---
TABLE_IDX = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
MODEL_IDX = int(sys.argv[2]) if len(sys.argv) >= 3 else 0

print(f"""
TABLE_IDX: {TABLE_IDX}
Model IDX: {MODEL_IDX}
""")

TABLE_NAME = ["train", "validation"][TABLE_IDX]

MODEL_PATH = ["llama3", "koesn/llama3-openbiollm-8b:q6_K", "gemma3", "alibayram/medgemma"][MODEL_IDX]
MODEL_NAME = ["llama3", "biollama3", "gemma3", "medgemma"][MODEL_IDX]

SAVED_FILE_NAME = f"{TABLE_NAME}_{MODEL_NAME}_zero_result.csv"

db_connector = DatabaseConnector(table_name=TABLE_NAME)
prompt_generator = PromptGenerator()
model_chat = ModelChat(MODEL_NAME, MODEL_PATH)
results = []

print(f"{SAVED_FILE_NAME} will be generated ... (Draft Results)")
# %%
# --- Draft Result Generation Loop ---
idx = 0
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

    idx += 1
    if idx % 100 == 0:
        df_result = pd.DataFrame(results)
        df_result.to_csv(f"outputs/generated_texts/plos/draft/{SAVED_FILE_NAME}", index=False)

# # %%
df_result = pd.DataFrame(results)
df_result.to_csv(f"outputs/generated_texts/plos/draft/{SAVED_FILE_NAME}", index=False)
print(f"{SAVED_FILE_NAME} will be generated ... (Draft Results)")
# %%