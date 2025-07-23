import joblib
import torch
import pickle
import pandas as pd
from model.wide_and_deep import WideAndDeep
from preprocessing.deep.stat_feature import get_normalize_scaler

# -------------------------------
# Configurable
# -------------------------------
STAGE_ID = "d4d"
TOP_K = 10
MODEL_PATH = "../wide_deep_model.pt"

# -------------------------------
# Load data
# -------------------------------
embeddings = torch.load("../graph/full_embeddings.pt")
node_mapping = torch.load("../graph/node_mapping_full.pt")
stage_id_to_idx = node_mapping["stage_id_to_idx"]
char_id_to_idx = node_mapping["character_id_to_idx"]

char_stat_dict = pickle.load(open("../char_stat_dict.pkl", "rb"))
char_wide_dict = torch.load("../preprocessing/wide/multi_hot/character_wide_feature_dict.pt")
enemy_wide_dict = torch.load("../preprocessing/wide/multi_hot/enemy_wide_feature_dict.pt")
stage_enemies_dict = torch.load("../preprocessing/wide/multi_hot/stage_enemies.pt")
enemy_stat_dict = joblib.load("../preprocessing/deep/enemy_stat_dict.pkl")


character_df = pd.read_csv("../db/characters.csv")
char_ids = character_df["id"].tolist()

# -------------------------------
# Helper: get stage enemy wide
# -------------------------------
def get_stage_enemy_wide(stage_id):
    enemy_ids = stage_enemies_dict.get(stage_id, [])
    if not enemy_ids:
        return torch.zeros_like(next(iter(enemy_wide_dict.values())))
    enemy_vecs = [enemy_wide_dict[eid] for eid in enemy_ids if eid in enemy_wide_dict]
    return torch.stack(enemy_vecs).mean(dim=0)

# -------------------------------
# Prepare input
# -------------------------------
deep_inputs, wide_inputs, char_id_list = [], [], []
stage_enemy_wide = get_stage_enemy_wide(STAGE_ID)
stage_emb = embeddings[stage_id_to_idx[STAGE_ID]]

if STAGE_ID in enemy_stat_dict:
    enemy_stat = torch.tensor(enemy_stat_dict[STAGE_ID], dtype=torch.float32)
else:
    enemy_stat = torch.zeros(len(next(iter(enemy_stat_dict.values()))))

for char_id in char_ids:
    if char_id not in char_id_to_idx or char_id not in char_stat_dict or char_id not in char_wide_dict:
        continue

    emb = embeddings[char_id_to_idx[char_id]]

    stat = torch.tensor(char_stat_dict[char_id], dtype=torch.float32)
    deep_vec = torch.cat([emb, stage_emb, stat, enemy_stat])

    wide_vec = torch.cat([char_wide_dict[char_id], stage_enemy_wide])

    deep_inputs.append(deep_vec)
    wide_inputs.append(wide_vec)
    char_id_list.append(char_id)

wide_x = torch.stack(wide_inputs)
deep_x = torch.stack(deep_inputs)

# -------------------------------
# Load model & Predict
# -------------------------------
model = WideAndDeep(wide_input_dim=wide_x.shape[1], deep_input_dim=deep_x.shape[1])
model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
model.eval()

with torch.no_grad():
    preds = model(wide_x, deep_x).squeeze()

# -------------------------------
# Rank & Output
# -------------------------------
topk_indices = torch.topk(preds, TOP_K).indices.tolist()
recommended_ids = [char_id_list[i] for i in topk_indices]
scores = preds[topk_indices].tolist()

print(f"✅ 추천 결과 (stage: {STAGE_ID})")
for rank, (cid, score) in enumerate(zip(recommended_ids, scores), 1):
    name = character_df[character_df["id"] == cid]["name"].values[0]
    print(f"{rank}. {name} (id: {cid}) - score: {score:.4f}")
