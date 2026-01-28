import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from lib.data.database_connector import DatabaseConnector
from lib.data.graph_generator import GraphGenerator
from lib.message.prompt_generator import PromptGenerator
from lib.model.model_chat import ModelChat
from lib.model.selector.gnn_selector import GNNSelector
from lib.utils.custom_loss import RougeRewardLoss

# --- Configuration ---
DB_PATH = 'dataset/plos.db'
TABLE_NAME = 'train'
MODEL_IDX = 2

LLM_MODEL_NAME = ["llama3", "biollama3", "gemma3", "medgemma"][MODEL_IDX]
LLM_MODEL_PATH = ["llama3", "koesn/llama3-openbiollm-8b:q6_K", "gemma3:270m", "alibayram/medgemma"][MODEL_IDX]
MODEL_SAVE_PATH = f'outputs/models/{LLM_MODEL_NAME}_gnn_selector.pth'
PLOT_SAVE_DIR = 'outputs/plots'

# Hyperparameters
HIDDEN_DIM = 64
EPOCHS = 5
LEARNING_RATE = 1e-4
MAX_SAMPLES = 100
TOP_N = 10
BATCH_SIZE = 4


def ensure_string(text_data):
    if isinstance(text_data, list):
        return " ".join([str(t) for t in text_data])
    return str(text_data)


def save_training_plots(loss_history, reward_history, model_name):
    """
    Eğitim metriklerini görselleştirir ve kaydeder.
    """
    if not os.path.exists(PLOT_SAVE_DIR):
        os.makedirs(PLOT_SAVE_DIR)

    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, 'b-o', label='Training Loss')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Reward Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs, reward_history, 'g-o', label='AvgRouge Reward')
    plt.title(f'{model_name} - Training Reward')
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(PLOT_SAVE_DIR, f"{model_name}_training_metrics.png")
    plt.savefig(plot_path)
    print(f"Training plots saved to: {plot_path}")
    plt.close()


def train():
    print(f"Initializing GNN Training (AvgRouge Reward) with Batch Size={BATCH_SIZE}...")

    # Initialize Components
    db = DatabaseConnector(db_path=DB_PATH, table_name=TABLE_NAME)

    # Draft CSV kontrolü
    draft_csv_path = f"outputs/generated_texts/plos/draft/train_{LLM_MODEL_NAME}_zero_result.csv"
    if not os.path.exists(draft_csv_path):
        print(f"Error: Draft CSV not found at {draft_csv_path}")
        return
    df_draft = pd.read_csv(draft_csv_path)

    graph_gen = GraphGenerator()
    prompt_gen = PromptGenerator()

    try:
        model_chat = ModelChat(LLM_MODEL_NAME, LLM_MODEL_PATH)
    except Exception as e:
        print(f"Warning: LLM Connection Failed. {e}")
        return

    rouge_loss_fn = RougeRewardLoss()  # Yeni kütüphane bazlı loss

    # Model Setup
    dummy_row = db[0]
    dummy_graph = graph_gen.generate_from_row(dummy_row)
    input_dim = dummy_graph.x.shape[1]

    model = GNNSelector(in_channels=input_dim, hidden_channels=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('outputs/models'):
        os.makedirs('outputs/models')

    model.train()

    # Metrics Storage
    epoch_loss_history = []
    epoch_reward_history = []

    for epoch in range(EPOCHS):
        total_reward = 0
        total_loss = 0
        count = 0
        batch_loss_acum = 0

        data_indices = range(min(len(db), MAX_SAMPLES))
        pbar = tqdm(data_indices, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        optimizer.zero_grad()

        for i in pbar:
            try:
                row = db[i]
                title = ensure_string(row['title'].values[0])
                abstract = ensure_string(row['abstract'].values[0])

                # Draft text alma
                draft_rows = df_draft.loc[df_draft['title'] == title]
                if not draft_rows.empty:
                    draft = draft_rows['generated_draft'].values[0]
                    # Draft çok kısaysa abstract kullan
                    draft = abstract if len(str(draft)) <= 10 else draft
                else:
                    draft = abstract  # Fallback

                gt_summary = ensure_string(row['summary'].values[0])
                sentences_list = [s for section in row['sections'][0] for s in section]

                # A. GNN Forward Pass
                graph = graph_gen.generate_from_row(row).to(device)
                probs = model(graph)

                # B. Selection
                num_nodes = probs.shape[0]
                k = min(TOP_N, num_nodes)
                if k == 0: continue

                top_k_scores, top_k_indices = torch.topk(probs.squeeze(), k=k)
                selected_indices_list = top_k_indices.detach().cpu().numpy().tolist()
                selected_sentences = [sentences_list[idx] for idx in selected_indices_list if idx < len(sentences_list)]

                # C. LLM Generation
                refine_prompt = prompt_gen.generate_refine_prompt(title, draft, selected_sentences)
                llm_output = model_chat.generate_zero_shot_answer(
                    refine_prompt['content'],
                    system_instruction=refine_prompt['instruction']
                )

                # D. Reward Calculation (AvgRouge)
                reward = rouge_loss_fn.get_reward(llm_output, gt_summary)

                # E. Loss Calculation
                log_probs = torch.log(top_k_scores + 1e-8)
                loss = rouge_loss_fn(log_probs, reward) / BATCH_SIZE

                # Backward
                loss.backward()

                batch_loss_acum += loss.item()
                total_reward += reward
                count += 1

                # F. Optimizer Step (Mini-Batch)
                if count % BATCH_SIZE == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += batch_loss_acum
                    batch_loss_acum = 0

                pbar.set_postfix({
                    "AvgRew": f"{total_reward / count:.3f}",
                    "CurrLoss": f"{loss.item() * BATCH_SIZE:.4f}"
                })

            except Exception as e:
                # print(f"Error at index {i}: {e}")
                continue

        # Epoch sonu kalan batch güncellemesi
        if count % BATCH_SIZE != 0:
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss_acum

        # Ortalamaları hesapla ve kaydet
        avg_reward = total_reward / count if count > 0 else 0
        # Loss'u batch normalization'ı geri alarak (x BATCH_SIZE ile değil, toplam loss üzerinden) hesaplıyoruz
        # Ancak yukarıda total_loss'a eklerken batch_loss_acum (yani loss/BATCH_SIZE) ekledik.
        # Bu yüzden gerçek total loss = total_loss * BATCH_SIZE.
        # Ortalama loss = (total_loss * BATCH_SIZE) / count
        avg_loss = (total_loss * BATCH_SIZE) / count if count > 0 else 0

        epoch_reward_history.append(avg_reward)
        epoch_loss_history.append(avg_loss)

        print(f"\nEpoch {epoch + 1} Done. Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Training Complete. Model saved to {MODEL_SAVE_PATH}")

    # Grafikleri çiz ve kaydet
    save_training_plots(epoch_loss_history, epoch_reward_history, LLM_MODEL_NAME)


if __name__ == "__main__":
    train()