from embeddings.build_graph import load_db_csv, make_node_index, make_edge_list, save_edge_index, save_node_mapping, build_graph
from train.train_lightgcn import  load_graph_data, get_num_nodes, get_ids, make_data, make_positive_edge, train_gnn
import yaml
from wide.multi_hot import build_wide_features

def load_config(path="../config/preprocessing.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # todo: db 변경 시에만 작동되도록

    # graph 생성
    config = load_config("../config/preprocessing.yaml")

    DATA_DIR = config["paths"]["db_dir"]
    GRAPH_DIR = config["paths"]["graph_dir"]
    WIDE_DIR = config["paths"]["wide_dir"]

    build_graph(DATA_DIR, GRAPH_DIR)

    # embedding 파일 생성
    edge_index, node_mapping = load_graph_data(GRAPH_DIR)

    char_ids, stage_ids = get_ids(node_mapping)

    num_nodes = get_num_nodes(edge_index)

    data = make_data(num_nodes, edge_index)

    positive_edges = make_positive_edge(char_ids, stage_ids, edge_index)

    train_gnn(GRAPH_DIR, data, positive_edges, num_nodes)



    # wide 전처리
    # todo: wide (on/off) config 파일로 관리
    build_wide_features(DATA_DIR, WIDE_DIR)

    # deep 전처리
    # todo: deep feature 별 (on/off) config 파일로 관리



    # 파일 하나로 통합

