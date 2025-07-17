# recommend_characters.py
import torch
import torch.nn.functional as F

# 데이터 로드
embeddings = torch.load("../graph/full_embeddings.pt")
node_mapping = torch.load("../graph/node_mapping_full.pt")

edge_index = torch.load("../graph/edge_index_full.pt")

# edge_index 내 최대 노드 번호
num_nodes = max(edge_index[0].max(), edge_index[1].max()).item() + 1
print("edge_index에 있는 최대 노드 번호:", num_nodes)

# node_mapping 내부 최대 인덱스 확인
all_indices = list(node_mapping["character_id_to_idx"].values()) + list(node_mapping["stage_id_to_idx"].values())
print("node_mapping 내 최대 인덱스:", max(all_indices))

# embeddings shape
#embeddings = torch.load("graph/character_embeddings.pt")
print("embedding 개수:", embeddings.shape[0])


# 추천 함수
def recommend_characters(stage_id: str, top_k=5):
    stage_idx = node_mapping["stage_id_to_idx"][stage_id]
    stage_emb = embeddings[stage_idx]

    char_id_to_idx = node_mapping["character_id_to_idx"]
    idx_to_char_id = {v: k for k, v in char_id_to_idx.items()}

    char_indices = list(char_id_to_idx.values())
    char_embs = embeddings[char_indices]

    # cosine similarity
    similarities = F.cosine_similarity(stage_emb.unsqueeze(0), char_embs)
    top_indices = similarities.topk(top_k).indices.tolist()
    recommended_ids = [idx_to_char_id[char_indices[i]] for i in top_indices]

    return recommended_ids

if __name__ == "__main__":
    ids = recommend_characters("rs", top_k=5)
    print(ids)