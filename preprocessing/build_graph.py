# build_graph_with_enemy.py
import pandas as pd
import torch
import os
import json

# 디렉토리 설정
DATA_DIR = "../db"
GRAPH_DIR = "../graph"
os.makedirs(GRAPH_DIR, exist_ok=True)

# 데이터 로드
characters = pd.read_csv(f"{DATA_DIR}/characters.csv")
char_props = pd.read_csv(f"{DATA_DIR}/character_properties.csv")
char_skills = pd.read_csv(f"{DATA_DIR}/character_skills.csv")
char_immus = pd.read_csv(f"{DATA_DIR}/character_immunities.csv")

enemies = pd.read_csv(f"{DATA_DIR}/enemies.csv")
enemy_props = pd.read_csv(f"{DATA_DIR}/enemy_properties.csv")
enemy_skills = pd.read_csv(f"{DATA_DIR}/enemy_skills.csv")
enemy_immus = pd.read_csv(f"{DATA_DIR}/enemy_immunities.csv")

skills = pd.read_csv(f"{DATA_DIR}/skills.csv")
properties = pd.read_csv(f"{DATA_DIR}/properties.csv")
immunities = pd.read_csv(f"{DATA_DIR}/immunities.csv")

stage_enemies = pd.read_csv(f"{DATA_DIR}/stage_enemies.csv")
stages = pd.read_csv(f"{DATA_DIR}/stages.csv")
# user_experience.csv 추가 로딩
user_exp = pd.read_csv(f"{DATA_DIR}/user_experiences.csv")

# 노드 인덱스 생성
character_ids = characters["id"].tolist()
property_ids = properties["id"].tolist()
skill_ids = skills["id"].tolist()
immunity_ids = immunities["id"].tolist()
enemy_ids = enemies["id"].tolist()
stage_ids = stages["id"].astype(str).unique()

num_characters = len(character_ids)
num_properties = len(property_ids)
num_skills = len(skill_ids)
num_immunities = len(immunity_ids)
num_enemies = len(enemy_ids)
num_stages = len(stage_ids)

property_offset = num_characters
skill_offset = property_offset + num_properties
immunity_offset = skill_offset + num_skills
enemy_offset = immunity_offset + num_immunities
stage_offset = enemy_offset + num_stages

character_id_to_idx = {cid: i for i, cid in enumerate(character_ids)}
property_id_to_idx = {pid: i + property_offset for i, pid in enumerate(property_ids)}
skill_id_to_idx = {sid: i + skill_offset for i, sid in enumerate(skill_ids)}
immunity_id_to_idx = {iid: i + immunity_offset for i, iid in enumerate(immunity_ids)}
enemy_id_to_idx = {eid: i + enemy_offset for i, eid in enumerate(enemy_ids)}
stage_id_to_idx = {sid: stage_offset + i for i, sid in enumerate(stage_ids)}

# 엣지 리스트 생성
src_list = []
dst_list = []

def add_bidirectional_edge(src, dst):
    src_list.extend([src, dst])
    dst_list.extend([dst, src])

# character ↔ property
for _, row in char_props.iterrows():
    c = character_id_to_idx.get(row["character_id"])
    p = property_id_to_idx.get(row["property_id"])
    if c is not None and p is not None:
        add_bidirectional_edge(c, p)

# character ↔ skill
for _, row in char_skills.iterrows():
    c = character_id_to_idx.get(row["character_id"])
    s = skill_id_to_idx.get(row["skill_id"])
    if c is not None and s is not None:
        add_bidirectional_edge(c, s)

# character ↔ immunity
for _, row in char_immus.iterrows():
    c = character_id_to_idx.get(row["character_id"])
    i = immunity_id_to_idx.get(row["immunity_id"])
    if c is not None and i is not None:
        add_bidirectional_edge(c, i)

# enemy ↔ property
for _, row in enemy_props.iterrows():
    e = enemy_id_to_idx.get(row["enemy_id"])
    p = property_id_to_idx.get(row["property_id"])
    if e is not None and p is not None:
        add_bidirectional_edge(e, p)

# enemy ↔ skill
for _, row in enemy_skills.iterrows():
    e = enemy_id_to_idx.get(row["enemy_id"])
    s = skill_id_to_idx.get(row["skill_id"])
    if e is not None and s is not None:
        add_bidirectional_edge(e, s)

# enemy ↔ immunity
for _, row in enemy_immus.iterrows():
    e = enemy_id_to_idx.get(row["enemy_id"])
    i = immunity_id_to_idx.get(row["immunity_id"])
    if e is not None and i is not None:
        add_bidirectional_edge(e, i)

# enemy ↔ stage
for _, row in stage_enemies.iterrows():
    e = enemy_id_to_idx.get(row["enemy_id"])
    stage_enemies["stage_id"] = stage_enemies["stage_id"].astype(str)
    s = stage_id_to_idx.get(row["stage_id"])
    if e is not None and s is not None:
        add_bidirectional_edge(e, s)



# character ↔ stage (from user experience)
for _, row in user_exp.iterrows():
    s = stage_id_to_idx.get(str(row["stage_id"]))  # stage_id는 str로 변환 필요
    try:
        characters_data = json.loads(row["characters_data"].replace("''", '"').replace("True", "true").replace("False", "false"))
    except:
        continue
    for ch in characters_data:
        c = character_id_to_idx.get(ch.get("character_id"))
        if c is not None and s is not None:
            add_bidirectional_edge(c, s)

# edge_index 저장
edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
torch.save(edge_index, f"{GRAPH_DIR}/edge_index_full.pt")

# 노드 인덱스 매핑 저장
node_mapping = {
    "character_id_to_idx": character_id_to_idx,
    "enemy_id_to_idx": enemy_id_to_idx,
    "property_id_to_idx": property_id_to_idx,
    "skill_id_to_idx": skill_id_to_idx,
    "immunity_id_to_idx": immunity_id_to_idx,
    "stage_id_to_idx": stage_id_to_idx
}
torch.save(node_mapping, f"{GRAPH_DIR}/node_mapping_full.pt")

# 메타 출력
print(f"Saved to {GRAPH_DIR}/edge_index_full.pt")
print(f"총 노드 수: {enemy_offset + num_enemies}")
print(f"총 엣지 수 (양방향 포함): {len(src_list)}")
