from embeddings.build_graph import load_db_csv, make_node_index, make_edge_list, save_edge_index, save_node_mapping
from train.train_lightgcn import  load_graph_data, get_num_nodes, get_ids, make_data, make_positive_edge, train_gnn


if __name__ == "__main__":
    # todo: db 변경 시에만 작동되도록

    # graph 생성
    DATA_DIR = "../db"
    GRAPH_DIR = "../graph"

    characters, properties, skills, immunities, enemies, stages, \
        char_props, char_skills, char_immus, \
        enemy_props, enemy_skills, enemy_immus, \
        stage_enemies, user_exp = load_db_csv(DATA_DIR)

    character_id_to_idx, property_id_to_idx, skill_id_to_idx, immunity_id_to_idx, enemy_id_to_idx, stage_id_to_idx = \
        make_node_index(characters, properties, skills, immunities, enemies, stages)

    src_list, dst_list = make_edge_list(char_props, char_skills, char_immus, enemy_props, enemy_skills, enemy_immus,
                                        character_id_to_idx, enemy_id_to_idx,
                                        property_id_to_idx, skill_id_to_idx, immunity_id_to_idx,
                                        stage_enemies, stage_id_to_idx, user_exp)
    save_edge_index(GRAPH_DIR, src_list, dst_list)

    save_node_mapping(GRAPH_DIR, character_id_to_idx, enemy_id_to_idx, property_id_to_idx, skill_id_to_idx, immunity_id_to_idx, stage_id_to_idx)

    # embedding 파일 생성
    edge_index, node_mapping = load_graph_data(GRAPH_DIR)

    char_ids, stage_ids = get_ids(node_mapping)

    num_nodes = get_num_nodes(edge_index)

    data = make_data(num_nodes, edge_index)

    positive_edges = make_positive_edge(char_ids, stage_ids, edge_index)

    train_gnn(GRAPH_DIR, data, positive_edges, num_nodes)



    # wide 전처리
    # todo: wide (on/off) config 파일로 관리


    # deep 전처리
    # todo: deep feature 별 (on/off) config 파일로 관리



    # 파일 하나로 통합

