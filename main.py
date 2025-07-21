import pickle

from model.deep_rec_model import DeepRecModel
import pandas as pd
import torch
import joblib
from train.train_deep import train_loop, set_data_train_val_by_stage
from preprocessing.stat_feature import get_normalize_scaler

if __name__ == "__main__":
    train_df = pd.read_csv('preprocessing/deep_train_with_stats.csv')

    embeddings = torch.load("graph/full_embeddings.pt")
    node_mapping = torch.load("graph/node_mapping_full.pt")
    stage_id_to_idx = node_mapping["stage_id_to_idx"]
    char_id_to_idx = node_mapping["character_id_to_idx"]

    # 필요한 파일들 로드
    character_df = pd.read_csv('db/characters.csv')
    scaler = joblib.load("preprocessing/stat_scaler.pkl")  # 정규화용 scaler 불러오기

    stat_features = [
        "hp", "atk1", "atk2", "atk3",
        "pre_atk1", "pre_atk2", "pre_atk3",
        "back_atk", "atk_freq",
        "range", "long_distance1", "long_distance2",
        "cost", "kb", "speed", "spawn"
    ]

    char_stat_dict = get_normalize_scaler(character_df)

    with open("char_stat_dict.pkl", "wb") as f:
        pickle.dump(char_stat_dict, f)

    train_loader, val_loader, test_loader = set_data_train_val_by_stage(train_df, embeddings, stage_id_to_idx, char_id_to_idx, char_stat_dict)

    model = DeepRecModel()

    train_loop(model, epochs=21, train_loader=train_loader, val_loader=val_loader)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).squeeze()
            all_preds.extend(preds.numpy())
            all_labels.extend(yb.numpy())


    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    thresholded_preds = [1 if p >= 0.5 else 0 for p in all_preds]

    print("Accuracy:", accuracy_score(all_labels, thresholded_preds))
    print("ROC AUC:", roc_auc_score(all_labels, all_preds))
    print("F1 Score:", f1_score(all_labels, thresholded_preds))
