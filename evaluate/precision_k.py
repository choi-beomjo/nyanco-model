import json

from test_model.test_deep import recommend_characters
import pandas as pd
import torch
from preprocessing.deep.char_feature import get_normalize_scaler


def f1_at_k(top_k, real_used):
    top_k_set = set(top_k)
    real_set = set(real_used)
    hit = len(top_k_set & real_set)

    precision = hit / len(top_k)
    recall = hit / len(real_set)

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


if __name__ == "__main__":

    #train_df = pd.read_csv('../preprocessing/deep_train_with_stats.csv')

    embeddings = torch.load("../graph/full_embeddings.pt")
    node_mapping = torch.load("../graph/node_mapping_full.pt")
    stage_id_to_idx = node_mapping["stage_id_to_idx"]
    char_id_to_idx = node_mapping["character_id_to_idx"]
    character_df = pd.read_csv('../db/characters.csv')

    # 🔸 character_id → base_id 매핑 딕셔너리
    charid_to_baseid = dict(zip(character_df["id"], character_df["base_id"]))

    char_stat_dict = get_normalize_scaler(character_df)

    model = torch.load("../deeprec_model.pt", weights_only=False)
    model.eval()

    user_experience = pd.read_csv('../db/user_experiences.csv')

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for _, row in user_experience.iterrows():
        stage_id = row["stage_id"]
        top5 = recommend_characters(stage_id=stage_id,
                                    embeddings=embeddings,
                                    node_mapping=node_mapping,
                                    character_df=character_df,
                                    char_stat_dict=char_stat_dict,
                                    model=model,
                                    top_k=5)

        char_ids_models = [char_id for rank, (char_id, score) in enumerate(top5, 1)]

        base_ids_models = {charid_to_baseid.get(cid, -1) for cid in char_ids_models}

        raw = row["characters_data"]  # 이건 문자열임
        clean_json_str = raw.replace('""', '"')  # CSV 이중 따옴표 처리 해제
        characters = json.loads(clean_json_str)

        character_ids_ex = [c["character_id"] for c in characters]
        base_ids_ex = {charid_to_baseid.get(cid, -2) for cid in character_ids_ex}

        precision, recall, f1 = f1_at_k(base_ids_models, base_ids_ex)
            # (f1_at_k(char_ids_models, character_ids_ex))

        # print(f"유저 경험 {row['id']} - precision {precision}")

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    print(f"Mean Precision@k:{sum(precision_scores)/len(precision_scores)}")
    print(f"Mean Recall@k:{sum(recall_scores) / len(recall_scores)}")
    print(f"Mean F1@k:{sum(f1_scores) / len(f1_scores)}")

