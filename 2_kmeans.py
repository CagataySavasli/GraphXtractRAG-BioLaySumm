import sys
import pandas as pd
from tqdm import tqdm

# Mevcut kütüphanelerini import ediyoruz
from lib.model.model_chat import ModelChat
from lib.data.database_connector import DatabaseConnector
from lib.message.prompt_generator import PromptGenerator

# Yeni yazdığımız KMeansRAG'i import ediyoruz (Dosya yoluna göre düzenle)
# Eğer ayrı dosya yapmadıysan yukarıdaki class'ı buraya da yapıştırabilirsin.
from lib.rag.kmeans_rag import KMeansRAG

# --- 1. SETTINGS and GENERATION ---
TABLE_IDX = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
MODEL_IDX = int(sys.argv[2]) if len(sys.argv) >= 3 else 0



print(f"""
--- K-MEANS REFINEMENT PROCESS ---
TABLE_IDX: {TABLE_IDX}
Model IDX: {MODEL_IDX}
""")

TABLE_NAME = ["train", "validation"][TABLE_IDX]
MODEL_PATH = ["llama3", "koesn/llama3-openbiollm-8b:q6_K", "gemma3", "alibayram/medgemma"][MODEL_IDX]
MODEL_NAME = ["llama3", "biollama3", "gemma3", "medgemma"][MODEL_IDX]

# Dosya ismini kmeans olarak güncelledik
SAVED_FILE_NAME = f"kmeans_{MODEL_NAME}_zero_result.csv"

# Nesne Başlatmaları
rag_engine = KMeansRAG(n=10)  # K-means motoru
db_connector = DatabaseConnector(table_name=TABLE_NAME)
prompt_generator = PromptGenerator()
model_chat = ModelChat(MODEL_NAME, MODEL_PATH)

# Draft sonuçlarını (Stage 1) çekiyoruz
draft_results = pd.read_csv(f"outputs/generated_texts/plos/draft/{TABLE_NAME}_{MODEL_NAME}_zero_result.csv")
results = []

print(f"{SAVED_FILE_NAME} will be generated ... (K-means Refine)")

# --- 2. Refine Step with K-means ---
idx = 0
for row in tqdm(db_connector, desc=f"K-means Refine with {MODEL_NAME}"):

    # --- STEP A: Data Preparation ---
    title = row['title'].values[0]
    original_summary = row['summary'].values[0]

    # Başlığa göre ilgili draft'ı bul
    pair_row = draft_results.loc[draft_results['title'] == title]
    if pair_row.empty:
        continue

    generated_draft_summary = pair_row['generated_draft'].values[0]

    # --- STEP B: K-means Sentence Selection ---
    # Burada Cosine Similarity yerine K-means merkezlerini kullanıyoruz
    selected_sentences = rag_engine.run(row)

    # --- STEP C: Prompt Generation & Model Call ---
    refine_prompt_dict = prompt_generator.generate_refine_prompt(
        title,
        generated_draft_summary,
        selected_sentences
    )

    refine_instruction = refine_prompt_dict['instruction']
    refine_content = refine_prompt_dict['content']

    # Modelden yanıt al (Minimum uzunluk kontrolü ile)
    final_lay_summary = ""
    retry_count = 0
    while len(final_lay_summary) < 50:
        retry_count += 1
        if retry_count > 2:
            break
        final_lay_summary = model_chat.generate_zero_shot_answer(refine_content, system_instruction=refine_instruction)

    # --- STEP D: Collect Results ---
    results.append({
        "title": title,
        "generated_draft": generated_draft_summary,
        "generated_final": final_lay_summary,
        "selected_sentences": selected_sentences,  # Analiz için ekledik
        "ground_truth": " ".join(original_summary) if isinstance(original_summary, list) else original_summary
    })

    # Ara kayıt
    idx += 1
    if idx % 100 == 0:
        pd.DataFrame(results).to_csv(f"outputs/generated_texts/plos/refine/{SAVED_FILE_NAME}", index=False)

# Final kayıt
df_result = pd.DataFrame(results)
df_result.to_csv(f"outputs/generated_texts/plos/refine/{SAVED_FILE_NAME}", index=False)
print(f"Process completed. Saved to: {SAVED_FILE_NAME}")