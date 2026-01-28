# %%
import sys
import os
import pandas as pd
from tqdm import tqdm
import torch

from lib.model.model_chat import ModelChat
from lib.data.database_connector import DatabaseConnector
from lib.rag.gnn_rag import GNNRAG  # Yeni yazdığımız modül
from lib.message.prompt_generator import PromptGenerator

# %%
# --- 1. SETTINGS and GENERATION ---
# Konsoldan argüman alma veya varsayılan değerler
TABLE_IDX = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
MODEL_IDX = int(sys.argv[2]) if len(sys.argv) >= 3 else 0

print(f"""
--- GNN BASED REFINE GENERATION ---
TABLE_IDX: {TABLE_IDX}
Model IDX: {MODEL_IDX}
""")

TABLE_NAME = ["train", "validation"][TABLE_IDX]

# LLM Ayarları (Özetleme Yapacak Model)
MODEL_PATH = ["llama3", "koesn/llama3-openbiollm-8b:q6_K", "gemma3", "alibayram/medgemma"][MODEL_IDX]
MODEL_NAME = ["llama3", "biollama3", "gemma3", "medgemma"][MODEL_IDX]

# Eğitilmiş GNN Modelinin Yolu (1_train_gnn.py çıktısı ile aynı olmalı)
# Eğer Mixed Objective kullandıysan isimlendirme farklı olabilir, burayı kontrol et.
TRAINED_GNN_PATH = f'outputs/models/{MODEL_NAME}_gnn_selector_mixed.pth'
# Eğer mixed değil normal train yaptıysan: f'outputs/models/{MODEL_NAME}_gnn_selector.pth'

if not os.path.exists(TRAINED_GNN_PATH):
    print(f"UYARI: {TRAINED_GNN_PATH} bulunamadı. Normal model deneniyor...")
    TRAINED_GNN_PATH = f'outputs/models/gnn_selector.pth'

SAVED_FILE_NAME = f"gnn_{MODEL_NAME}_zero_result.csv"

print(f"""
--- GNN BASED REFINE GENERATION ---
TABLE_NAME: {TABLE_NAME}
Model: {MODEL_NAME}
""")

# --- 2. INITIALIZE COMPONENTS ---

# A. GNN RAG Modülü
try:
    rag_inner = GNNRAG(model_path=TRAINED_GNN_PATH, top_n=10)
except FileNotFoundError as e:
    print(f"Kritik Hata: {e}")
    sys.exit(1)

# B. Diğer Bileşenler
db_connector = DatabaseConnector(table_name=TABLE_NAME)
prompt_generator = PromptGenerator()
model_chat = ModelChat(MODEL_NAME, MODEL_PATH)

# C. Draft Sonuçlarını Yükle (Input for Refine Stage)
draft_csv_path = f"outputs/generated_texts/plos/draft/{TABLE_NAME}_{MODEL_NAME}_zero_result.csv"
if not os.path.exists(draft_csv_path):
    print(f"Hata: Draft dosyası bulunamadı -> {draft_csv_path}")
    sys.exit(1)

draft_results = pd.read_csv(draft_csv_path)
results = []

print(f"{SAVED_FILE_NAME} üretiliyor... (Refine Results)")

# --- 3. MAIN LOOP ---
# Veri seti üzerinde dön
for idx in tqdm(range(len(db_connector)), desc="GNN Refine Process"):
    try:
        # Veriyi çek
        row_df = db_connector[idx]  # DataFrame döner
        row = row_df.iloc[0]  # Seriye çevir

        title = row['title']
        original_summary = row['summary']

        # Draft Metni Bul
        draft_row = draft_results.loc[draft_results['title'] == title]
        if not draft_row.empty:
            generated_draft_summary = str(draft_row['generated_draft'].values[0])
        else:
            # Eğer draft yoksa abstract kullan (Fallback)
            generated_draft_summary = str(row['abstract'])

        # --- STEP A: Sentence Selection (GNN) ---
        # GNN modelini kullanarak en önemli cümleleri seç
        selected_sentences = rag_inner.run(row)

        if not selected_sentences:
            print(f"Warning: No sentences selected for {title}")
            selected_sentences = [generated_draft_summary]  # Fallback

        # --- STEP B: Refine Step (LLM) ---

        # 1. Prompt Oluştur
        refine_prompt_dict = prompt_generator.generate_refine_prompt(
            title,
            generated_draft_summary,  # Pre-Generated Draft
            selected_sentences  # Selected Sentences by GNN
        )

        refine_instruction = refine_prompt_dict['instruction']
        refine_content = refine_prompt_dict['content']

        # 2. LLM'e Sor (Final Summary)
        final_lay_summary = ""
        count_final = 0
        while len(final_lay_summary) < 50:  # Min length check
            count_final += 1
            if count_final > 2:
                final_lay_summary = generated_draft_summary  # Fail safe
                break

            final_lay_summary = model_chat.generate_zero_shot_answer(
                refine_content,
                system_instruction=refine_instruction
            )

        # --- STEP C: Save Result Row ---
        results.append({
            "title": title,
            "generated_draft": generated_draft_summary,
            "generated_final": final_lay_summary,
            "ground_truth": " ".join(original_summary) if isinstance(original_summary, list) else original_summary
        })

        # Her 100 adımda bir kaydet (Checkpoint)
        if len(results) % 100 == 0:
            output_dir = "outputs/generated_texts/plos/refine"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            df_result = pd.DataFrame(results)
            df_result.to_csv(f"{output_dir}/{SAVED_FILE_NAME}", index=False)

    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        continue

# --- 4. FINAL SAVE ---
output_dir = "outputs/generated_texts/plos/refine"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_result = pd.DataFrame(results)
df_result.to_csv(f"{output_dir}/{SAVED_FILE_NAME}", index=False)
print(f"Process Completed. Saved to {output_dir}/{SAVED_FILE_NAME}")