import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch.cuda.amp import autocast, GradScaler
import gc
import numpy as np

from lib.data.database_connector import DatabaseConnector
from lib.data.graph_generator import GraphGenerator
from lib.model.selector.gnn_selector import GNNSelector

# --- AYARLAR ---
DB_PATH = 'dataset/plos.db'
TABLE_NAME = 'train'
OUTPUT_FILE = 'dataset/all_labels_train_relevance.pt'
MODEL_SAVE_PATH = 'outputs/models/gnn_selector_hybrid_relevance.pth'
PLOT_SAVE_DIR = 'outputs/plots'

# --- MEMORY STRATEJİSİ ---
BATCH_SIZE = 1
GRAD_ACCUMULATION = 16
CPU_OFFLOAD_THRESHOLD = 350

EPOCHS = 20
LEARNING_RATE = 1e-4
HIDDEN_DIM = 64
NUM_WORKERS = 4


class FullGnnDataset(Dataset):
    def __init__(self, db_path, table_name, labels_file):
        super().__init__()
        self.db = DatabaseConnector(db_path=db_path, table_name=table_name)
        self.graph_gen = GraphGenerator()

        print(f"Labellar yükleniyor...")
        self.labels_dict = torch.load(labels_file)
        self.valid_indices = sorted(list(self.labels_dict.keys()))

        max_db_idx = len(self.db) - 1
        self.valid_indices = [idx for idx in self.valid_indices if idx <= max_db_idx]
        print(f"Dataset hazır. Veri Sayısı: {len(self.valid_indices)}")
        print("NOT: Max Sentence Limiti YOK. Tüm makale işlenecek.")

    def len(self):
        return len(self.valid_indices)

    def get(self, idx):
        real_idx = self.valid_indices[idx]
        row_df = self.db[real_idx]
        row = row_df.iloc[0]

        data = self.graph_gen.generate_from_row(row)
        y = self.labels_dict[real_idx]
        y = torch.tensor(y).unsqueeze(1)

        if data.x.shape[0] != y.shape[0]:
            min_len = min(data.x.shape[0], y.shape[0])
            if data.x.shape[0] > y.shape[0]:
                new_y = torch.zeros((data.x.shape[0], 1), dtype=torch.float)
                new_y[:min_len] = y[:min_len]
                data.y = new_y
            else:
                data.y = y[:data.x.shape[0]]
        else:
            data.y = y

        return data


def train_step_safe(model, batch, criterion, device_gpu, scaler):
    num_nodes = batch.x.shape[0]

    if num_nodes > CPU_OFFLOAD_THRESHOLD:
        device = torch.device('cpu')
        use_amp = False
    else:
        device = device_gpu
        use_amp = True

    try:
        batch = batch.to(device)
        original_device = next(model.parameters()).device
        if device != original_device:
            model.to(device)

        if use_amp:
            with autocast():
                out = model(batch)
                probs = torch.sigmoid(out)
                loss = criterion(probs, batch.y)
        else:
            out = model(batch)
            probs = torch.sigmoid(out)
            loss = criterion(probs, batch.y)

        loss_scaled = loss / GRAD_ACCUMULATION

        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if device != original_device:
            model.to(original_device)

        return loss.item(), "GPU" if device == device_gpu else "CPU"

    except RuntimeError as e:
        if "out of memory" in str(e) and device == device_gpu:
            torch.cuda.empty_cache()
            return train_step_safe(model, batch.to('cpu'), criterion, torch.device('cpu'), scaler)
        else:
            raise e


def train():
    gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Master Device: {gpu_device}")

    dataset = FullGnnDataset(DB_PATH, TABLE_NAME, OUTPUT_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    sample_data = dataset[0]
    model = GNNSelector(in_channels=sample_data.x.shape[1], hidden_channels=HIDDEN_DIM).to(gpu_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH))

    model.train()

    # Plot için loss değerlerini tutacağımız liste
    epoch_losses = []

    for epoch in range(EPOCHS):
        total_loss = 0
        batch_count = 0
        optimizer.zero_grad()
        cpu_usage_count = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for i, batch in enumerate(pbar):
            loss_val, used_device = train_step_safe(model, batch, criterion, gpu_device, scaler)

            if used_device == "CPU":
                cpu_usage_count += 1

            if (i + 1) % GRAD_ACCUMULATION == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            total_loss += loss_val
            batch_count += 1

            pbar.set_postfix({"Loss": f"{loss_val:.4f}"})

        if len(loader) % GRAD_ACCUMULATION != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch + 1} Done. Avg Loss: {avg_loss:.5f}. CPU Offload: {cpu_usage_count}")

        # O anki epoch'un ortalama kaybını listeye ekliyoruz
        epoch_losses.append(avg_loss)

        # --- CHECKPOINT KAYDETME ---
        # 1. Her epoch için ayrı kaydet: gnn_selector_hybrid_epoch_1.pth
        root, ext = os.path.splitext(MODEL_SAVE_PATH)
        epoch_path = f"{root}_epoch_{epoch + 1}{ext}"
        torch.save(model.state_dict(), epoch_path)
        print(f"Checkpoint saved: {epoch_path}")

        # 2. Ana dosyayı güncelle (En son hali)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # --- EĞİTİM SONRASI PLOT OLUŞTURMA VE KAYDETME ---
    print("Grafik oluşturuluyor ve kaydediliyor...")
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_SAVE_DIR, 'gnn_selector_train_loss_relevance.png')

    plt.figure(figsize=(8, 4.5))
    epochs_range = range(1, EPOCHS + 1)

    # Çizgi ve noktaları resimdeki gibi ekliyoruz
    plt.plot(epochs_range, epoch_losses, marker='o', linestyle='-')

    plt.title('Epoch vs Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')

    # X eksenini 2, 4, 6... 20 olacak şekilde ayarlıyoruz
    plt.xticks(range(2, EPOCHS + 1, 2))

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot başarıyla kaydedildi: {plot_path}")


if __name__ == "__main__":
    train()