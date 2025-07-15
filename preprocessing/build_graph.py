import pandas as pd
import torch

# ----------------------
# Load ../db
# ----------------------
characters = pd.read_csv("../db/characters.csv")
char_props = pd.read_csv("../db/character_properties.csv")
char_skills = pd.read_csv("../db/character_skills.csv")
char_immus = pd.read_csv("../db/character_immunities.csv")
skills = pd.read_csv("../db/skills.csv")
properties = pd.read_csv("../db/properties.csv")
immunities = pd.read_csv("../db/immunities.csv")

# ----------------------
# Index mapping
# ----------------------
character_ids = characters["id"].tolist()
property_ids = properties["id"].tolist()
skill_ids = skills["id"].tolist()
immunity_ids = immunities["id"].tolist()

num_characters = len(character_ids)
num_properties = len(property_ids)
num_skills = len(skill_ids)
num_immunities = len(immunity_ids)

property_offset = num_characters
skill_offset = property_offset + num_properties
immunity_offset = skill_offset + num_skills

# 인덱스 매핑
character_id_to_idx = {cid: idx for idx, cid in enumerate(character_ids)}
property_id_to_idx = {pid: idx + property_offset for idx, pid in enumerate(property_ids)}
skill_id_to_idx = {sid: idx + skill_offset for idx, sid in enumerate(skill_ids)}
immunity_id_to_idx = {iid: idx + immunity_offset for idx, iid in enumerate(immunity_ids)}

# ----------------------
# Build edges
# ----------------------
src_list = []
dst_list = []

# character ↔ property
for _, row in char_props.iterrows():
    c_idx = character_id_to_idx.get(row["character_id"])
    p_idx = property_id_to_idx.get(row["property_id"])
    if c_idx is not None and p_idx is not None:
        src_list += [c_idx, p_idx]
        dst_list += [p_idx, c_idx]

# character ↔ skill
for _, row in char_skills.iterrows():
    c_idx = character_id_to_idx.get(row["character_id"])
    s_idx = skill_id_to_idx.get(row["skill_id"])
    if c_idx is not None and s_idx is not None:
        src_list += [c_idx, s_idx]
        dst_list += [s_idx, c_idx]

# character ↔ immunity
for _, row in char_immus.iterrows():
    c_idx = character_id_to_idx.get(row["character_id"])
    i_idx = immunity_id_to_idx.get(row["immunity_id"])
    if c_idx is not None and i_idx is not None:
        src_list += [c_idx, i_idx]
        dst_list += [i_idx, c_idx]

# ----------------------
# Create edge_index
# ----------------------
edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

# ----------------------
# Save for inspection
# ----------------------
print(f"Total nodes: {num_characters + num_properties + num_skills + num_immunities}")
print(f"Edge index shape: {edge_index.shape}")
print(f"Sample edges: \n{edge_index[:, :5]}")


print(f"Max node index: {edge_index.max()}")
print(f"Total expected nodes: {num_characters + num_properties + num_skills + num_immunities}")

from collections import Counter

edge_type_counter = Counter()

for src, dst in zip(src_list, dst_list):
    if (
        (src < num_characters and property_offset <= dst < skill_offset) or
        (dst < num_characters and property_offset <= src < skill_offset)
    ):
        edge_type_counter['character-property'] += 1
    elif (
        (src < num_characters and skill_offset <= dst < immunity_offset) or
        (dst < num_characters and skill_offset <= src < immunity_offset)
    ):
        edge_type_counter['character-skill'] += 1
    elif (
        (src < num_characters and dst >= immunity_offset) or
        (dst < num_characters and src >= immunity_offset)
    ):
        edge_type_counter['character-immunity'] += 1


print(edge_type_counter)

target_character_id = 7
target_idx = character_id_to_idx[target_character_id]

connected = edge_index[1][edge_index[0] == target_idx]
print(f"Character {target_character_id} ({target_idx}) is connected to nodes: {connected.tolist()}")

# 역방향 인덱스 맵
idx_to_property_id = {v: k for k, v in property_id_to_idx.items()}
idx_to_skill_id = {v: k for k, v in skill_id_to_idx.items()}
idx_to_immunity_id = {v: k for k, v in immunity_id_to_idx.items()}

# 어떤 노드인지 확인하는 함수
def resolve_node_type(node_idx):
    if node_idx >= immunity_offset:
        return 'immunity', idx_to_immunity_id.get(node_idx)
    elif node_idx >= skill_offset:
        return 'skill', idx_to_skill_id.get(node_idx)
    elif node_idx >= property_offset:
        return 'property', idx_to_property_id.get(node_idx)
    else:
        return 'character', node_idx  # 내부 character_idx

# 테스트 출력
for idx in [1802, 1836]:
    node_type, original_id = resolve_node_type(idx)
    print(f"Connected to {node_type} (original ID: {original_id})")

