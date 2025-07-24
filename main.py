import pickle
import pandas as pd
import torch
import joblib
from model.wide_and_deep import WideAndDeep
from train.train_wide_deep import train_loop, set_data_train_val_by_stage_with_char_enemy_wide
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from util.load_config import load_config

if __name__ == "__main__":

    config = load_config("./config/train.yaml")

    paths = config["paths"]

    train_df_path = paths["train_df"]
    embeddings_path = paths["embeddings"]
    node_mapping_path = paths["node_mapping"]
    char_stat_path = paths["char_stat_dict"]
    enemy_stat_path = paths["enemy_stat_dict"]

    char_wide_path = paths["character_wide"]
    enemy_wide_path = paths["enemy_wide"]
    stage_enemies_path = paths["stage_enemies"]


    # ---------------------------
    # 1. 데이터 로드
    # ---------------------------
    train_df = pd.read_csv(train_df_path)
    embeddings = torch.load(embeddings_path)
    node_mapping = torch.load(node_mapping_path)
    stage_id_to_idx = node_mapping["stage_id_to_idx"]
    char_id_to_idx = node_mapping["character_id_to_idx"]

    # ---------------------------
    # 2. 정규화된 스탯 딕셔너리
    # ---------------------------
    char_stat_dict = joblib.load(char_stat_path)
    enemy_stat_dict = joblib.load(enemy_stat_path)

    # ---------------------------
    # 3. Wide feature 불러오기
    # ---------------------------
    char_wide_dict = torch.load(char_wide_path)  # {char_id: tensor}
    enemy_wide_dict = torch.load(enemy_wide_path)     # {enemy_id: tensor}
    stage_enemies_dict = torch.load(stage_enemies_path)         # {stage_id: [enemy_id, ...]}

    # ---------------------------
    # 4. 데이터셋 구성
    # ---------------------------

    hyper_params = config["hyperparameters"]

    epochs = hyper_params["epochs"]
    learning_rate = hyper_params["learning_rate"]
    batch_size = hyper_params["batch_size"]

    train_loader, val_loader, test_loader, wide_input_dim, deep_input_dim = set_data_train_val_by_stage_with_char_enemy_wide(
        train_df,
        embeddings,
        stage_id_to_idx,
        char_id_to_idx,
        char_stat_dict,
        char_wide_dict,
        enemy_wide_dict,
        stage_enemies_dict,
        enemy_stat_dict,
        batch_size
    )

    # ---------------------------
    # 5. 모델 생성 및 학습
    # ---------------------------

    model = WideAndDeep(wide_input_dim=wide_input_dim, deep_input_dim=deep_input_dim)
    train_loop(model, train_loader=train_loader, val_loader=val_loader, epochs=epochs, learning_rate=learning_rate)

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
