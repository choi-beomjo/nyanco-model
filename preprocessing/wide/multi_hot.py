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
    # ✅ 0. 전체 ID 목록 미리 불러오기
    character_ids = pd.read_csv("../../db/characters.csv")["id"].unique()
    enemy_ids = pd.read_csv("../../db/enemies.csv")["id"].unique()

    # ✅ 1. character-skill
    char_skill_df = pd.read_csv("../../db/character_skills.csv")
    char_skill_df, char_skill_tensor = create_multi_hot_matrix(char_skill_df, "character_id", "skill_id", "skill")
    char_skill_df = char_skill_df.reindex(character_ids, fill_value=0).astype(int)

    # ✅ 2. character-property
    char_prop_df = pd.read_csv("../../db/character_properties.csv")
    char_prop_df, char_prop_tensor = create_multi_hot_matrix(char_prop_df, "character_id", "property_id", "prop")
    char_prop_df = char_prop_df.reindex(character_ids, fill_value=0).astype(int)

    # ✅ 3. enemy-skill
    enemy_skill_df = pd.read_csv("../../db/enemy_skills.csv")
    enemy_skill_df, enemy_skill_tensor = create_multi_hot_matrix(enemy_skill_df, "enemy_id", "skill_id", "skill")
    enemy_skill_df = enemy_skill_df.reindex(enemy_ids, fill_value=0).astype(int)

    # ✅ 4. enemy-property
    enemy_prop_df = pd.read_csv("../../db/enemy_properties.csv")
    enemy_prop_df, enemy_prop_tensor = create_multi_hot_matrix(enemy_prop_df, "enemy_id", "property_id", "prop")
    enemy_prop_df = enemy_prop_df.reindex(enemy_ids, fill_value=0).astype(int)

    # ✅ 5. enemy-immunity
    enemy_imm_df = pd.read_csv("../../db/enemy_immunities.csv")
    enemy_imm_df, enemy_imm_tensor = create_multi_hot_matrix(enemy_imm_df, "enemy_id", "immunity_id", "imm")
    enemy_imm_df = enemy_imm_df.reindex(enemy_ids, fill_value=0).astype(int)

    # ✅ 6. character-immunity
    char_imm_df = pd.read_csv("../../db/character_immunities.csv")
    char_imm_df, char_imm_tensor = create_multi_hot_matrix(char_imm_df, "character_id", "immunity_id", "imm")
    char_imm_df = char_imm_df.reindex(character_ids, fill_value=0).astype(int)

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

    # ✅ 7. character-wide 통합 (outer join)
    char_dfs = [char_skill_df, char_prop_df, char_imm_df]
    char_wide_df = char_dfs[0].copy()
    for df in char_dfs[1:]:
        char_wide_df = char_wide_df.merge(df, left_index=True, right_index=True, how="outer")
    char_wide_df = char_wide_df.fillna(0).astype(int)

    # ✅ 8. enemy-wide 통합
    enemy_dfs = [enemy_skill_df, enemy_prop_df, enemy_imm_df]
    enemy_wide_df = enemy_dfs[0].copy()
    for df in enemy_dfs[1:]:
        enemy_wide_df = enemy_wide_df.merge(df, left_index=True, right_index=True, how="outer")
    enemy_wide_df = enemy_wide_df.fillna(0).astype(int)

    # ✅ 9. 텐서 및 dict 저장
    char_wide_tensor = torch.tensor(char_wide_df.values).float()
    enemy_wide_tensor = torch.tensor(enemy_wide_df.values).float()

    char_wide_dict = {int(cid): char_wide_tensor[i] for i, cid in enumerate(char_wide_df.index)}
    enemy_wide_dict = {int(eid): enemy_wide_tensor[i] for i, eid in enumerate(enemy_wide_df.index)}

    torch.save(char_wide_dict, "multi_hot/character_wide_feature_dict.pt")
    torch.save(enemy_wide_dict, "multi_hot/enemy_wide_feature_dict.pt")

    # 스테이지-적 매핑 파일 로드
    df = pd.read_csv("../../db/stage_enemies.csv")

    # groupby로 딕셔너리 생성
    stage_enemies_dict = (
        df.groupby("stage_id")["enemy_id"]
        .apply(list)
        .to_dict()
    )

    # 저장
    torch.save(stage_enemies_dict, "multi_hot/stage_enemies.pt")

    print("✅ 모든 wide feature 저장 완료")

