import os
import torch
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from lib.data.database_connector import DatabaseConnector

# --- AYARLAR ---
DB_PATH = 'dataset/plos.db'
TABLE_NAME = 'train'
OUTPUT_FILE = 'dataset/all_labels_train.pt'  # Tek dosya
MAX_SAMPLES = 25000


def calculate_heuristic_labels(sentences, ground_truth, scorer):
    scores = []
    if not ground_truth or not isinstance(ground_truth, str):
        return torch.zeros(len(sentences), dtype=torch.float).view(-1, 1)

    for sentence in sentences:
        if not sentence or len(str(sentence).strip()) < 5:
            scores.append(0.0)
            continue
        try:
            s = scorer.score(ground_truth, str(sentence))
            avg_score = (s['rouge1'].fmeasure + s['rouge2'].fmeasure + s['rougeLsum'].fmeasure) / 3.0
            scores.append(avg_score)
        except:
            scores.append(0.0)

    scores_np = np.array(scores, dtype=np.float32)

    if scores_np.max() > 0:
        min_val = scores_np.min()
        max_val = scores_np.max()
        if max_val - min_val > 1e-6:
            scores_np = (scores_np - min_val) / (max_val - min_val)

    return torch.tensor(scores_np, dtype=torch.float).view(-1, 1)


def generate_labels():
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    print(f"Labels calculating for {TABLE_NAME} (In-Memory)...")

    db = DatabaseConnector(db_path=DB_PATH, table_name=TABLE_NAME)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    # Tüm labelları burada tutacağız: {index: tensor}
    all_labels = {}

    indices = range(min(len(db), MAX_SAMPLES))

    for i in tqdm(indices):
        try:
            row_df = db[i]
            row = row_df.iloc[0]

            gt_summary = row['summary']
            if isinstance(gt_summary, list):
                gt_summary = " ".join([str(s) for s in gt_summary])

            sentences = [s for section in row['sections'] for s in section]

            if not sentences: continue

            y = calculate_heuristic_labels(sentences, gt_summary, scorer)

            # Sözlüğe ekle
            all_labels[i] = y

        except Exception as e:
            continue

    # Tek seferde diske kaydet
    print(f"Hesaplama bitti. {len(all_labels)} adet label '{OUTPUT_FILE}' dosyasına kaydediliyor...")
    torch.save(all_labels, OUTPUT_FILE)
    print("Kaydedildi.")


if __name__ == "__main__":
    generate_labels()