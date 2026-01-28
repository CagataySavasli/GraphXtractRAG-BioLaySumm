import pandas as pd
import os
from lib.utils.result_calculator import evaluate

# Konfigürasyonlar
MODELS = ['gemma3'] #["llama3", "biollama3", "gemma3", "medgemma"]
RAG_APPROACHES = ['gnn']#["similarity", "pagerank"]
PROMPT_STRATEGY = "zero"

# Sonuçları toplamak için sözlükler
abstract_results = []
similarity_refine_results = []

for model in MODELS:
    for approach in RAG_APPROACHES:
        FILE_NAME = f"{approach}_{model}_{PROMPT_STRATEGY}_result.csv"
        FILE_PATH = os.path.join("outputs/generated_texts/plos/refine", FILE_NAME)

        if not os.path.exists(FILE_PATH):
            print(f"Uyarı: {FILE_PATH} bulunamadı, atlanıyor.")
            continue

        # Veriyi yükle ve eksikleri doldur
        df_results = pd.read_csv(FILE_PATH)
        print(FILE_NAME)
        print(f"{df_results['generated_final'].isna().sum()} - {len(df_results)}")
        df_results['generated_final'] = df_results['generated_final'].fillna(df_results['generated_draft'])

        preds_draft = df_results['generated_draft'].tolist()
        preds_final = df_results['generated_final'].tolist()
        references = df_results['ground_truth'].tolist()

        # Metrikleri hesapla
        print(f"Hesaplanıyor: {model} - {approach}")
        draft_scores = evaluate(preds_draft, references)
        final_scores = evaluate(preds_final, references)

        # CSV olarak kaydet
        output_df = pd.DataFrame([draft_scores, final_scores], index=['draft', 'final'])
        output_dir = "outputs/scores"
        os.makedirs(output_dir, exist_ok=True)
        output_df.to_csv(f"{output_dir}/{model}_{approach}_{PROMPT_STRATEGY}_results.csv")

        # Tablo verisi için sakla (Sadece ilk approach döngüsünde draft skorlarını alıyoruz)
        if approach == "similarity":
            # Baseline (Draft) tablosu için
            draft_row = {"Model": model, **draft_scores}
            abstract_results.append(draft_row)

            # Similarity Refine tablosu için (Final skorları)
            refine_row = {"Model": model, **final_scores}
            similarity_refine_results.append(refine_row)


# --- LaTeX Tablo Üretimi ---

def generate_latex_table(data, caption, label):
    # Model isimlerini görsel tabloya uygun hale getir
    display_names = {
        "llama3": "LlaMa 3",
        "biollama3": "OpenBioLLMs",
        "gemma3": "Gemma 3",
        "medgemma": "MedGemma"
    }

    latex_str = r"\begin{table}[H]" + "\n"
    latex_str += r"\centering" + "\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += f"\\label{{{label}}}\n"
    latex_str += r"\resizebox{\textwidth}{!}{" + "\n"
    latex_str += r"\begin{tabular}{@{}lccccccc@{}}" + "\n"
    latex_str += r"\toprule" + "\n"
    latex_str += r"\textbf{GenAI Model} & \textbf{R1} & \textbf{R2} & \textbf{RL} & \textbf{BERT} & \textbf{FKGL} & \textbf{DCRS} & \textbf{CLI} \\ \midrule" + "\n"

    for row in data:
        m_name = display_names.get(row['Model'], row['Model'])
        latex_str += f"{m_name} & {row['ROUGE1']:.4f} & {row['ROUGE2']:.4f} & {row['ROUGEL']:.4f} & {row['BERTScore']:.4f} & {row['FKGL']:.4f} & {row['DCRS']:.4f} & {row['CLI']:.4f} \\\\\n"

    latex_str += r"\bottomrule" + "\n"
    latex_str += r"\end{tabular}" + "\n"
    latex_str += "}\n"
    latex_str += r"\end{table}"
    return latex_str


print("\n" + "=" * 20 + " LATEX KODLARI " + "=" * 20 + "\n")
print(generate_latex_table(abstract_results, "Baseline Performance: Only Abstract (Draft) Results",
                           "tab:abstract-results"))
print("\n")
print(generate_latex_table(similarity_refine_results, "Performance Analysis: Refine with Similarity Approach",
                           "tab:similarity-refine-results"))
