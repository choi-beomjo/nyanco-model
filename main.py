import pickle
import pandas as pd
import torch
import joblib
from model.wide_and_deep import WideAndDeep
from preprocessing.deep.char_feature import get_normalize_scaler
from train.train_wide_deep import train_loop, set_data_train_val_by_stage_with_char_enemy_wide
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

if __name__ == "__main__":
    # ---------------------------
    # 1. 데이터 로드
    # ---------------------------
    train_df = pd.read_csv('preprocessing/deep_train_with_stats.csv')
    embeddings = torch.load("graph/full_embeddings.pt")
    node_mapping = torch.load("graph/node_mapping_full.pt")
    stage_id_to_idx = node_mapping["stage_id_to_idx"]
    char_id_to_idx = node_mapping["character_id_to_idx"]
    character_df = pd.read_csv('db/characters.csv')
    scaler = joblib.load("preprocessing/stat_scaler.pkl")

    # ---------------------------
    # 2. 정규화된 스탯 딕셔너리
    # ---------------------------
    char_stat_dict = get_normalize_scaler(character_df)
    with open("char_stat_dict.pkl", "wb") as f:
        pickle.dump(char_stat_dict, f)


    enemy_stat_dict = joblib.load("preprocessing/deep/enemy_stat_dict.pkl")

    # ---------------------------
    # 3. Wide feature 불러오기
    # ---------------------------
    char_wide_dict = torch.load("preprocessing/wide/multi_hot/character_wide_feature_dict.pt")  # {char_id: tensor}
    enemy_wide_dict = torch.load("preprocessing/wide/multi_hot/enemy_wide_feature_dict.pt")     # {enemy_id: tensor}
    stage_enemies_dict = torch.load("preprocessing/wide/multi_hot/stage_enemies.pt")         # {stage_id: [enemy_id, ...]}

    # ---------------------------
    # 4. 데이터셋 구성
    # ---------------------------
    train_loader, val_loader, test_loader, wide_input_dim, deep_input_dim = set_data_train_val_by_stage_with_char_enemy_wide(
        train_df,
        embeddings,
        stage_id_to_idx,
        char_id_to_idx,
        char_stat_dict,
        char_wide_dict,
        enemy_wide_dict,
        stage_enemies_dict,
        enemy_stat_dict
    )

    # ---------------------------
    # 5. 모델 생성 및 학습
    # ---------------------------
    model = WideAndDeep(wide_input_dim=wide_input_dim, deep_input_dim=deep_input_dim)
    train_loop(model, train_loader=train_loader, val_loader=val_loader, epochs=40)

    # ---------------------------
    # 6. 테스트
    # ---------------------------
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for wide_x, deep_x, y in test_loader:
            pred = model(wide_x, deep_x).squeeze()
            all_preds.extend(pred.numpy())
            all_labels.extend(y.numpy())

    thresholded_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    print("Accuracy:", accuracy_score(all_labels, thresholded_preds))
    print("ROC AUC:", roc_auc_score(all_labels, all_preds))
    print("F1 Score:", f1_score(all_labels, thresholded_preds))
