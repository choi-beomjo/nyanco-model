import joblib

stat_features = [
    "hp", "atk1", "atk2", "atk3",
    "pre_atk1", "pre_atk2", "pre_atk3",
    "back_atk", "atk_freq",
    "range", "long_distance1", "long_distance2",
    "cost", "kb", "speed", "spawn"
]

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

scaler = MinMaxScaler()

character_df = pd.read_csv('../db/characters.csv')

normalized_stats = scaler.fit_transform(character_df[stat_features])

# character_df에는 character_id 컬럼이 있다고 가정
character_id_list = character_df["id"].tolist()

# character_id -> normalized vector
char_stat_dict = {
    char_id: normalized_stats[i]
    for i, char_id in enumerate(character_id_list)
}

train_df = pd.read_csv("stage_character_train.csv")

# stat vector 붙이기
stat_features_df = pd.DataFrame(
    train_df["character_id"].map(char_stat_dict).tolist(),
    columns=[f"stat_{f}" for f in stat_features]
)

# 붙이기
train_df = pd.concat([train_df, stat_features_df], axis=1)

train_df.to_csv("deep_train_with_stats.csv", index=False)
joblib.dump(scaler, "stat_scaler.pkl")

