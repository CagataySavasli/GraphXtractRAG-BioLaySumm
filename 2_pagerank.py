# %%
import sys
import pandas as pd
from tqdm import tqdm

from lib.model.model_chat import ModelChat
from lib.data.database_connector import DatabaseConnector
from lib.data.graph_generator import GraphGenerator
from lib.rag.pagerank import PageRankRAG
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

SAVED_FILE_NAME = f"pagerank_{MODEL_NAME}_zero_result.csv"

rag_inner = PageRankRAG()
graph_generator = GraphGenerator()
db_connector = DatabaseConnector(table_name=TABLE_NAME)
prompt_generator = PromptGenerator()
model_chat = ModelChat(MODEL_NAME, MODEL_PATH)
draft_results = pd.read_csv(f"outputs/generated_texts/plos/draft/{TABLE_NAME}_{MODEL_NAME}_zero_result.csv")
results = []

print(f"{SAVED_FILE_NAME} will be generated ... (Refine Results)")
# %%
# --- Draft Result Generation Loop ---
idx = 0
for row in tqdm(db_connector, desc="Makaleler Özetleniyor"):

    # --- STEP A: Get Data and Find Pair Draft---
    title = row['title'].values[0]
    original_summary = row['summary'].values[0] # Ground Truth

    pair_row = draft_results.loc[draft_results['title'] == title]
    generated_draft_summary = pair_row['generated_draft'].values[0]
    abstract = pair_row['abstract'].values[0]

    # --- STEP A: Sentence Selection  ---
    try:
        # 1. Generate Graph
        graph_data = graph_generator.generate_from_row(row)

        # 2. Select Sentences with PageRank
        selected_sentences = rag_inner.run(row, graph_data)

    except Exception as e:
        print(f"Graph/PageRank Error (Title: {title}): {e}")
        selected_sentences = [] # Liste Error Cases


    # --- STEP B: Refine Step ---
    # 1. Generate Prompt

    refine_prompt_dict = prompt_generator.generate_refine_prompt(
        title,
        generated_draft_summary, # Pre-Generated Draft
        selected_sentences       # Selected Sentences with PageRank
    )

    refine_instruction = refine_prompt_dict['instruction']
    refine_content = refine_prompt_dict['content']

    # 2. Ask Model (Generate Draft Summary)
    final_lay_summary = ""
    count_final = 0
    while len(final_lay_summary) < 50:  # min length check
        count_final += 1
        if count_final > 2:
            final_lay_summary = ""
            break
        final_lay_summary = model_chat.generate_zero_shot_answer(refine_content, system_instruction=refine_instruction)

    # --- STEP C: Save Answers ---
    results.append({
        "title": title,
        "generated_draft": generated_draft_summary,
        "generated_final": final_lay_summary,
        "ground_truth": " ".join(original_summary) if isinstance(original_summary, list) else original_summary
    })
    idx += 1
    if idx % 100 == 0:
        df_result = pd.DataFrame(results)
        df_result.to_csv(f"outputs/generated_texts/plos/refine/{SAVED_FILE_NAME}", index=False)
# %%
df_result = pd.DataFrame(results)
df_result.to_csv(f"outputs/generated_texts/plos/refine/{SAVED_FILE_NAME}", index=False)
print(f"{SAVED_FILE_NAME} will be generated ... (Refine Results)")
# %%
