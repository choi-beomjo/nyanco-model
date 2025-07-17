import pandas as pd
import json

df = pd.read_csv("../db/user_experiences.csv")

positive_pairs = set()

for _, row in df.iterrows():
    stage_id = row["stage_id"]
    try:
        characters = json.loads(row["characters_data"].replace('""', '"'))  # 문자열 파싱
        for entry in characters:
            char_id = entry["character_id"]
            positive_pairs.add((stage_id, char_id))
    except Exception as e:
        print(f"Error parsing row {row['id']}: {e}")

from collections import defaultdict

stage_to_pos_chars = defaultdict(set)
for stage_id, char_id in positive_pairs:
    stage_to_pos_chars[stage_id].add(char_id)

import random

all_char_ids = set(range(1803))  # 모든 캐릭터 ID 목록
samples = []

for stage_id, pos_chars in stage_to_pos_chars.items():
    # Positive 샘플
    for cid in pos_chars:
        samples.append({"stage_id": stage_id, "character_id": cid, "label": 1})

    # Negative 샘플
    neg_cands = list(all_char_ids - pos_chars)
    neg_sampled = random.sample(neg_cands, k=len(pos_chars))  # 1:1 비율
    for cid in neg_sampled:
        samples.append({"stage_id": stage_id, "character_id": cid, "label": 0})


train_df = pd.DataFrame(samples)
train_df.to_csv("stage_character_train.csv", index=False)
