import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import joblib
from torch.utils.data import TensorDataset, DataLoader


def build_X_y(df, embeddings, stage_id_to_idx, char_id_to_idx, char_stat_dict):
    X = []
    for _, row in df.iterrows():
        stage_id = row["stage_id"]
        character_id = row["character_id"]

        stage_emb = embeddings[stage_id_to_idx[stage_id]].numpy()
        char_emb = embeddings[char_id_to_idx[character_id]].numpy()
        stat_vec = char_stat_dict[character_id]

        X.append(np.concatenate([stage_emb, char_emb, stat_vec]))

    X = np.array(X, dtype=np.float32)
    y = df["label"].values.astype(np.float32)
    return X, y

# 실행

def set_data_train_val_by_stage(train_df, embeddings, stage_id_to_idx, char_id_to_idx, char_stat_dict):
    # 스테이지 아이디를 기준으로 데이터 분류
    all_stage_ids = train_df["stage_id"].unique()

    # 1차: train + temp
    stage_train, stage_temp = train_test_split(
        all_stage_ids, test_size=0.3, random_state=42
    )

    # 2차: validation + test
    stage_val, stage_test = train_test_split(
        stage_temp, test_size=0.5, random_state=42
    )

    df_train = train_df[train_df["stage_id"].isin(stage_train)]
    df_val = train_df[train_df["stage_id"].isin(stage_val)]
    df_test = train_df[train_df["stage_id"].isin(stage_test)]


    X_train, y_train = build_X_y(df_train, embeddings, stage_id_to_idx, char_id_to_idx, char_stat_dict)
    X_val, y_val = build_X_y(df_val, embeddings, stage_id_to_idx, char_id_to_idx, char_stat_dict)
    X_test, y_test = build_X_y(df_test, embeddings, stage_id_to_idx, char_id_to_idx, char_stat_dict)


    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    return train_loader, val_loader, test_loader


def train_loop(model, epochs, train_loader, val_loader):
# model = DeepRecModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(1, epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 검증 단계
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).squeeze()
                loss = loss_fn(pred, yb)
                val_loss += loss.item()
                predicted = (pred >= 0.5).float()
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

        acc = correct / total

        print(f"[Epoch {epoch}] Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")

    torch.save(model, "deeprec_model.pt")