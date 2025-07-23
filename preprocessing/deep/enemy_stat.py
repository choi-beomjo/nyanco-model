import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


def build_stage_enemy_stat_dict(stage_enemies_df, enemy_stat_dict, stat_features):
    from collections import defaultdict
    import numpy as np

    stage_to_enemy_stats = defaultdict(list)

    for _, row in stage_enemies_df.iterrows():
        stage_id = row["stage_id"]
        enemy_id = row["enemy_id"]

        if enemy_id in enemy_stat_dict:
            stage_to_enemy_stats[stage_id].append(enemy_stat_dict[enemy_id])

    # 평균 벡터로 정리
    stage_enemy_stat_vec = {
        sid: np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(len(stat_features))
        for sid, vecs in stage_to_enemy_stats.items()
    }

    return stage_enemy_stat_vec


if __name__ == "__main__":

    scaler = MinMaxScaler()

    stat_features = [
        "hp", "atk1", "atk2", "atk3",
        "pre_atk1", "pre_atk2", "pre_atk3",
        "back_atk", "atk_freq",
        "range", "long_distance1", "long_distance2",
        "money", "kb", "speed", "tba"
    ]

    enemy_df = pd.read_csv('../../db/enemies.csv')
    normalized_stats = scaler.fit_transform(enemy_df[stat_features])

    joblib.dump(scaler, "../enemy_scaler.pkl")

    enemy_id_list = enemy_df["id"].tolist()

    # enemy_id -> normalized vector
    enemy_stat_dict = {
        char_id: normalized_stats[i]
        for i, char_id in enumerate(enemy_id_list)
    }

    joblib.dump(enemy_stat_dict, "../enemy_stat_dict.pkl")

    df = pd.read_csv('../../db/stage_enemies.csv')

    stage_enemy_stat_vec = build_stage_enemy_stat_dict(df, enemy_stat_dict, stat_features)
    joblib.dump(stage_enemy_stat_vec, "../stage_enemy_stat_vec.pkl")
