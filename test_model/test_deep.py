import torch
from train.train_deep import set_data_train_val_by_stage
from preprocessing.stat_feature import get_normalize_scaler
import pandas as pd
import numpy as np


def recommend_characters(stage_id, top_k=5):
    stage_vec = embeddings[stage_id_to_idx[stage_id]].numpy()  # (64,)

    candidates = []
    for char_id in character_df["id"]:
        if char_id not in char_id_to_idx or char_id not in char_stat_dict:
            continue

        char_vec = embeddings[char_id_to_idx[char_id]].numpy()         # (64,)
        stat_vec = char_stat_dict[char_id]                             # (16,)
        input_vec = np.concatenate([stage_vec, char_vec, stat_vec])    # (144,)

        with torch.no_grad():
            pred = model(torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)).item()

        candidates.append((char_id, pred))

    # 정렬 후 Top-K 추출
    top_k_chars = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
    return top_k_chars


if __name__ == "__main__":

    train_df = pd.read_csv('../preprocessing/deep_train_with_stats.csv')

    embeddings = torch.load("../graph/full_embeddings.pt")
    node_mapping = torch.load("../graph/node_mapping_full.pt")
    stage_id_to_idx = node_mapping["stage_id_to_idx"]
    char_id_to_idx = node_mapping["character_id_to_idx"]
    character_df = pd.read_csv('../db/characters.csv')

    char_stat_dict = get_normalize_scaler(character_df)

    model = torch.load("../deeprec_model.pt", weights_only=False)
    model.eval()

    all_preds = []
    all_labels = []

    _, _, test_loader = set_data_train_val_by_stage(train_df, embeddings, stage_id_to_idx, char_id_to_idx, char_stat_dict)

    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).squeeze()
            all_preds.extend(preds.numpy())
            all_labels.extend(yb.numpy())


    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    thresholded_preds = [1 if p >= 0.5 else 0 for p in all_preds]

    print("Accuracy:", accuracy_score(all_labels, thresholded_preds))
    print("ROC AUC:", roc_auc_score(all_labels, all_preds))
    print("F1 Score:", f1_score(all_labels, thresholded_preds))

    char_name = character_df.set_index("id")["name"].to_dict()

    top5 = recommend_characters(stage_id="d4d", top_k=5)
    for rank, (char_id, score) in enumerate(top5, 1):
        print(f"{rank}. {char_name[char_id]} (ID {char_id}) - Score: {score:.4f}")

