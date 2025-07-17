import numpy as np
import pandas as pd
import torch


train_df = pd.read_csv('../preprocessing/deep_train_with_stats.csv')

embeddings = torch.load("../graph/full_embeddings.pt")
node_mapping = torch.load("../graph/node_mapping_full.pt")

stage_id_to_idx = node_mapping["stage_id_to_idx"]
char_id_to_idx = node_mapping["character_id_to_idx"]

import joblib

# 필요한 파일들 로드
character_df = pd.read_csv('../db/characters.csv')
scaler = joblib.load("../preprocessing/stat_scaler.pkl")  # 정규화용 scaler 불러오기

stat_features = [
    "hp", "atk1", "atk2", "atk3",
    "pre_atk1", "pre_atk2", "pre_atk3",
    "back_atk", "atk_freq",
    "range", "long_distance1", "long_distance2",
    "cost", "kb", "speed", "spawn"
]

# 정규화
normalized_stats = scaler.transform(character_df[stat_features])

# character_id ↔ normalized vector 매핑
character_id_list = character_df["id"].tolist()
char_stat_dict = {
    char_id: normalized_stats[i]
    for i, char_id in enumerate(character_id_list)
}


deep_input_vectors = []

for _, row in train_df.iterrows():
    stage_id = row["stage_id"]
    character_id = row["character_id"]

    stage_emb = embeddings[stage_id_to_idx[stage_id]]  # (64,)
    char_emb = embeddings[char_id_to_idx[character_id]]  # (64,)
    stat_vec = char_stat_dict[character_id]  # (16,)

    # 순서 중요: [stage, character, stat]
    input_vec = np.concatenate([stage_emb, char_emb, stat_vec])  # (144,)
    deep_input_vectors.append(input_vec)

X = np.array(deep_input_vectors)
y = train_df["label"].values

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

import torch
import torch.nn as nn

class DeepRecModel(nn.Module):
    def __init__(self, input_dim=144):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

model = DeepRecModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch}] Loss: {total_loss:.4f}")
