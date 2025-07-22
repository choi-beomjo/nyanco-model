import torch
from preprocessing.deep.stat_feature import get_normalize_scaler
import pandas as pd
import numpy as np


def recommend_characters(stage_id, embeddings, node_mapping, character_df, char_stat_dict, model, top_k=5):

    stage_id_to_idx = node_mapping["stage_id_to_idx"]
    char_id_to_idx = node_mapping["character_id_to_idx"]

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

    #train_df = pd.read_csv('../preprocessing/deep_train_with_stats.csv')

    embeddings = torch.load("../graph/full_embeddings.pt")
    node_mapping = torch.load("../graph/node_mapping_full.pt")
    stage_id_to_idx = node_mapping["stage_id_to_idx"]
    char_id_to_idx = node_mapping["character_id_to_idx"]
    character_df = pd.read_csv('../db/characters.csv')

    char_stat_dict = get_normalize_scaler(character_df)

    model = torch.load("../deeprec_model.pt", weights_only=False)
    model.eval()


    char_name = character_df.set_index("id")["name"].to_dict()

    top5 = recommend_characters(stage_id="d4d",
                                embeddings=embeddings,
                                node_mapping=node_mapping,
                                character_df=character_df,
                                char_stat_dict=char_stat_dict,
                                model=model,
                                top_k=10)
    for rank, (char_id, score) in enumerate(top5, 1):
        print(f"{rank}. {char_name[char_id]} (ID {char_id}) - Score: {score:.4f}")

