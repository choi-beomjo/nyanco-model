import pandas as pd
import json
import torch


# 디렉토리 설정
DATA_DIR = "../db"
GRAPH_DIR = "../graph"

# CSV 파일 경로
user_exp_path = f"{DATA_DIR}/user_experiences.csv"
stages_path = f"{DATA_DIR}/stages.csv"
characters_path = f"{DATA_DIR}/characters.csv"

# 데이터 불러오기
user_exp = pd.read_csv(user_exp_path)
stages = pd.read_csv(stages_path)
characters = pd.read_csv(characters_path)

# 고유 ID 추출
character_ids = characters["id"].unique()
stage_ids = stages["id"].astype(str).unique()  # stage_id가 문자열인 경우 있음

# 인덱스 매핑 (LightGCN에서는 사용자 ↔ 아이템 형태로 만들어야 하므로)
character_id_to_idx = {cid: i for i, cid in enumerate(character_ids)}
stage_id_to_idx = {sid: i + len(character_ids) for i, sid in enumerate(stage_ids)}  # character 다음부터 시작

src_list, dst_list = [], []

# 유저 경험 기반 엣지 생성
for _, row in user_exp.iterrows():
    try:
        characters_data = json.loads(row["characters_data"].replace('""', '"'))
        stage_id = str(row["stage_id"])
        s_idx = stage_id_to_idx.get(stage_id)
        if s_idx is None:
            continue

        for char in characters_data:
            c_idx = character_id_to_idx.get(char["character_id"])
            if c_idx is None:
                continue
            # 양방향 엣지 추가
            src_list += [c_idx, s_idx]
            dst_list += [s_idx, c_idx]
    except json.JSONDecodeError:
        continue

# Tensor로 변환 및 저장
edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
torch.save(edge_index, f"{GRAPH_DIR}/edge_index_char_stage.pt")

print("총 엣지 수:", edge_index.shape[1])
print("총 노드 수:", len(character_id_to_idx) + len(stage_id_to_idx))
