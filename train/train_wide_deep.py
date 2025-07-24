# train/train_wide_deep.py
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def get_stage_enemy_wide(stage_id, enemy_wide_dict, stage_enemies_dict):
    enemy_ids = stage_enemies_dict.get(stage_id, [])
    if not enemy_ids:
        return torch.zeros_like(next(iter(enemy_wide_dict.values())))
    enemy_vecs = [enemy_wide_dict[eid] for eid in enemy_ids if eid in enemy_wide_dict]
    return torch.stack(enemy_vecs).mean(dim=0)

def set_data_train_val_by_stage_with_char_enemy_wide(
    df, embeddings, stage_id_to_idx, char_id_to_idx, stat_dict,
    char_wide_dict, enemy_wide_dict, stage_enemies_dict, enemy_stat_dict,
    batch_size=64
):
    wide_inputs, deep_inputs, labels = [], [], []

    for _, row in df.iterrows():
        char_id = row["character_id"]
        stage_id = row["stage_id"]
        label = row["label"]

        # deep input
        emb = embeddings[char_id_to_idx[char_id]]
        stage_emb = embeddings[stage_id_to_idx[stage_id]]
        stat = torch.tensor(stat_dict[char_id], dtype=torch.float32)

        if stage_id in enemy_stat_dict:
            enemy_stat = torch.tensor(enemy_stat_dict[stage_id], dtype=torch.float32)
        else:
            enemy_stat = torch.zeros(len(stat))  # fallback

        deep = torch.cat([emb, stage_emb, stat, enemy_stat])

        # wide input = char_wide + enemy_wide
        char_vec = char_wide_dict[char_id]
        enemy_vec = get_stage_enemy_wide(stage_id, enemy_wide_dict, stage_enemies_dict)
        wide = torch.cat([char_vec, enemy_vec])

        wide_inputs.append(wide)
        deep_inputs.append(deep)
        labels.append(label)

    wide_x = torch.stack(wide_inputs)
    deep_x = torch.stack(deep_inputs)
    y = torch.tensor(labels, dtype=torch.float32)

    w_train, w_temp, d_train, d_temp, y_train, y_temp = train_test_split(wide_x, deep_x, y, test_size=0.3)
    w_val, w_test, d_val, d_test, y_val, y_test = train_test_split(w_temp, d_temp, y_temp, test_size=0.5)

    train_loader = DataLoader(TensorDataset(w_train, d_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(w_val, d_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(w_test, d_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, w_train.shape[1], d_train.shape[1]


def train_loop(model, train_loader, val_loader, epochs=10, learning_rate=1e-3):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for wide_x, deep_x, y in train_loader:
            pred = model(wide_x, deep_x).squeeze()
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for wide_x, deep_x, y in val_loader:
                pred = model(wide_x, deep_x).squeeze()
                val_preds.extend(pred.numpy())
                val_labels.extend(y.numpy())
        val_preds_binary = [1 if p >= 0.5 else 0 for p in val_preds]
        acc = accuracy_score(val_labels, val_preds_binary)
        print(f"Validation Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "wide_deep_model.pt")
