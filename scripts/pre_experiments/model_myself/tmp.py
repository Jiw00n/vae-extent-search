from tvm.auto_scheduler.feature import get_per_store_features_from_file, get_per_store_features_from_measure_pairs
from tvm.auto_scheduler.workload_registry import register_workload, serialize_args, get_func_name, workload_key_to_dag
from make_dataset import load_and_register_tasks
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class regression(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation="relu"):
        super(regression, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, latent_dim),
            nn.Linear(latent_dim, 1)
        )



    def forward(self, x):
        pred_cost = self.fc(x)
        return pred_cost

load_and_register_tasks()
raw_features, raw_normalized_throughputs, task_ids, min_latency = get_per_store_features_from_file("/root/work/tenset/dataset/measure_records/k80/([0c9a5ba46ffc5e1a9e5641018527117f,4,7,7,160,1,1,160,960,1,1,1,960,4,7,7,960],cuda).json", 10000)


raw_features = torch.tensor(raw_features.tolist(), dtype=torch.float32)
raw_normalized_throughputs = torch.tensor(raw_normalized_throughputs, dtype=torch.float32).unsqueeze(1)

masks = torch.zeros_like(raw_normalized_throughputs)

# for idx, t in enumerate(raw_normalized_throughputs):
#     if t != 8.908700362966152e-15:
#         masks[idx] = True




        


# features shape : (4000,)
# features[0] shape : (n_bufs, feature_dim)
# feature normalization
feature = (features - features.mean()) / features.std()

# seed 고정
torch.manual_seed(42)
np.random.seed(42)

HIDDEN_DIM = 256
LATENT_DIM = 128
BATCH_SIZE = 128
EPOCHS = 1000
LEARNING_RATE = 1e-5

# --- 3. DataLoader 준비 (마스크 포함) ---
from torch.utils.data import random_split, DataLoader, TensorDataset

# 전체 데이터셋
dataset = TensorDataset(feature, costs)

# train : val = 8 : 2 (2960 : 740 정도)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader 준비
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

# --- 4. 모델, 옵티마이저 초기화 (이전과 동일) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = regression(input_dim=max_len, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, activation="relu").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. 학습 루프 (마스킹된 손실 계산) ---
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        loss = F.l1_loss(pred, y_batch, reduction='sum')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = F.l1_loss(pred, y_batch, reduction='sum')
            val_loss += loss.item()
    val_loss /= len(val_dataset)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
