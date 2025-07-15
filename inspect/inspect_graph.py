# inspect_graph.py
import torch

# 파일 경로
EDGE_FILE = "../graph/edge_index_full.pt"
MAPPING_FILE = "../graph/node_mapping_full.pt"

# 1. 데이터 로드
edge_index = torch.load(EDGE_FILE)
node_mapping = torch.load(MAPPING_FILE)

src, dst = edge_index[0], edge_index[1]

# 2. 기본 정보 출력
print(f"총 노드 수: {max(src.max(), dst.max()).item() + 1}")
print(f"총 엣지 수 (양방향 포함): {src.size(0)}")

# 3. 노드 타입 범위 계산
def get_offset(name):
    return min(node_mapping[f"{name}_id_to_idx"].values())

def get_max_idx(mapping):
    return max(mapping.values())

offsets = {
    "character": (get_offset("character"), get_max_idx(node_mapping["character_id_to_idx"])),
    "property": (get_offset("property"), get_max_idx(node_mapping["property_id_to_idx"])),
    "skill": (get_offset("skill"), get_max_idx(node_mapping["skill_id_to_idx"])),
    "immunity": (get_offset("immunity"), get_max_idx(node_mapping["immunity_id_to_idx"])),
    "enemy": (get_offset("enemy"), get_max_idx(node_mapping["enemy_id_to_idx"])),
}

print("\n[노드 타입별 인덱스 범위]")
for name, (start, end) in offsets.items():
    print(f"{name:<10}: {start:>4} ~ {end:>4}")

# 4. 예시: 특정 캐릭터 ID가 연결된 노드 출력
CHARACTER_ID = 7
c_idx = node_mapping["character_id_to_idx"].get(CHARACTER_ID)

if c_idx is not None:
    connected = dst[src == c_idx].tolist()
    print(f"\nCharacter {CHARACTER_ID} (idx={c_idx}) is connected to {len(connected)} nodes:")
    print(connected[:20], "...")  # 너무 많으면 일부만 출력
else:
    print(f"Character {CHARACTER_ID}는 node_mapping에 없음")

# 5. 노드별 연결 수 요약
from collections import Counter

counts = Counter()
for s in src.tolist():
    counts[s] += 1

print("\n상위 5개 연결 노드 (연결 수 기준):")
for node_idx, cnt in counts.most_common(5):
    print(f"노드 {node_idx} : {cnt} 회 연결")
