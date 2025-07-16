import torch
from torch_geometric.nn import LightGCN
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from tqdm import tqdm

# 데이터 로드
edge_index = torch.load("../graph/edge_index_full.pt")
node_mapping = torch.load("../graph/node_mapping_full.pt")

char_ids = list(node_mapping["character_id_to_idx"].values())
stage_ids = list(node_mapping["stage_id_to_idx"].values())

num_nodes = max(edge_index[0].max(), edge_index[1].max()).item() + 1
data = Data(edge_index=edge_index, num_nodes=num_nodes)

# character ↔ stage 관계만 positive edge로 필터링
positive_edges = []
for c_id in char_ids:
    neighbors = edge_index[1][edge_index[0] == c_id]
    for s_id in neighbors:
        if s_id.item() in stage_ids:
            positive_edges.append([c_id, s_id.item()])
positive_edges = torch.tensor(positive_edges, dtype=torch.long).T  # shape [2, num_pos]

# 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
positive_edges = positive_edges.to(device)

model = LightGCN(num_nodes=num_nodes, embedding_dim=64, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습 루프
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()

    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=num_nodes,
        num_neg_samples=positive_edges.size(1)
    )

    edge_label_index = torch.cat([positive_edges, neg_edge_index], dim=1)
    edge_label = torch.cat([
        torch.ones(positive_edges.size(1), device=device),
        torch.zeros(neg_edge_index.size(1), device=device)
    ])

    pred = model(data.edge_index, edge_label_index)
    loss = model.recommendation_loss(pred[:positive_edges.size(1)], pred[positive_edges.size(1):], node_id=edge_label_index.unique())
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

# 최종 임베딩 저장
model.eval()
with torch.no_grad():
    final_embeddings = model.get_embedding(data.edge_index)

torch.save(final_embeddings, "../graph/character_embeddings.pt")
print("Character embeddings saved.")
