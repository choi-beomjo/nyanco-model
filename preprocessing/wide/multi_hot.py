import pandas as pd
import torch


def create_multi_hot_matrix(df, id_col, feature_col, prefix):
    """
    df: DataFrame with id_col (e.g., character_id) and feature_col (e.g., skill_id)
    id_col: 고유 ID 기준 (e.g., character_id, enemy_id)
    feature_col: 속성, 스킬, 무효 등 카테고리
    prefix: 컬럼 prefix (e.g., 'skill', 'property')

    returns: multi-hot dataframe, torch tensor
    """
    pivot_df = df.assign(value=1).pivot_table(
        index=id_col,
        columns=feature_col,
        values="value",
        fill_value=0
    )
    pivot_df.columns = [f"{prefix}_{col}" for col in pivot_df.columns]
    return pivot_df, torch.tensor(pivot_df.values).float()



if __name__ == "__main__":
    # 1. character-skill
    char_skill_df = pd.read_csv("../../db/character_skills.csv")  # "character_id","skill_id"
    char_skill_df, char_skill_tensor = create_multi_hot_matrix(char_skill_df, "character_id", "skill_id", "skill")

    # 2. character-property
    char_prop_df = pd.read_csv("../../db/character_properties.csv")  # "character_id","property_id"
    char_prop_df, char_prop_tensor = create_multi_hot_matrix(char_prop_df, "character_id", "property_id", "prop")

    # 3. enemy-skill
    enemy_skill_df = pd.read_csv("../../db/enemy_skills.csv")  # "enemy_id","skill_id"
    enemy_skill_df, enemy_skill_tensor = create_multi_hot_matrix(enemy_skill_df, "enemy_id", "skill_id", "skill")

    # 4. enemy-property
    enemy_prop_df = pd.read_csv("../../db/enemy_properties.csv")  # "enemy_id","property_id"
    enemy_prop_df, enemy_prop_tensor = create_multi_hot_matrix(enemy_prop_df, "enemy_id", "property_id", "prop")

    # 5. enemy-immunity
    enemy_imm_df = pd.read_csv("../../db/enemy_immunities.csv")  # "enemy_id","immunity_id"
    enemy_imm_df, enemy_imm_tensor = create_multi_hot_matrix(enemy_imm_df, "enemy_id", "immunity_id", "imm")

    # 6. character-immunity
    char_imm_df = pd.read_csv("../../db/character_immunities.csv")  # "enemy_id","immunity_id"
    char_imm_df, char_imm_tensor = create_multi_hot_matrix(char_imm_df, "character_id", "immunity_id", "imm")

    # os.makedirs("multi_hot", exist_ok=True)

    char_skill_df.to_csv("multi_hot/character_skills_wide.csv")
    torch.save(char_skill_tensor, "multi_hot/character_skills.pt")

    char_prop_df.to_csv("multi_hot/character_properties_wide.csv")
    torch.save(char_prop_tensor, "multi_hot/character_properties.pt")

    enemy_skill_df.to_csv("multi_hot/enemy_skills_wide.csv")
    torch.save(enemy_skill_tensor, "multi_hot/enemy_skills.pt")

    enemy_prop_df.to_csv("multi_hot/enemy_properties_wide.csv")
    torch.save(enemy_prop_tensor, "multi_hot/enemy_properties.pt")

    enemy_imm_df.to_csv("multi_hot/enemy_immunities_wide.csv")
    torch.save(enemy_imm_tensor, "multi_hot/enemy_immunities.pt")

    char_imm_df.to_csv("multi_hot/character_immunities_wide.csv")
    torch.save(char_imm_tensor, "multi_hot/character_immunities.pt")

