import pandas as pd
import torch
from torch_geometric.data import Data  # Import Data
import json
import numpy as np
from sklearn.model_selection import train_test_split

def fetch_labels():
    # Load target file
    target = pd.read_csv('./twitch/ENGB/musae_ENGB_target_edited.csv')

    # Create label tensor
    labels = target['mature'].astype(int).values

    return labels

def load_twitch_dataset():
    # Load edges file
    edges = pd.read_csv('./twitch/ENGB/musae_ENGB_edges_edited.csv', sep=',')
    # print(edges)

    print(edges.columns)
    print(edges.head())

    # Ensure columns are integers
    edges['Source'] = pd.to_numeric(edges['Source'], errors='coerce').fillna(0).astype(int)
    edges['Target'] = pd.to_numeric(edges['Target'], errors='coerce').fillna(0).astype(int)

    # Convert to edge index tensor
    edge_index = torch.tensor(edges[['Source', 'Target']].values.T, dtype=torch.long)

    # Create graph data object
    data = Data(edge_index=edge_index)
    print(data)

    # Load node features
    with open('./twitch/ENGB/musae_ENGB_features.json') as f:
        features = json.load(f)

    # Convert features to a matrix
    node_features = np.zeros((len(features), max(max(f) for f in features.values()) + 1))
    for node, feats in features.items():
        node_features[int(node), feats] = 1  # One-hot encoding of features

    # Convert to tensor
    x = torch.tensor(node_features, dtype=torch.float)
    data.x = x
    print(data)

    labels = fetch_labels()

    y = torch.tensor(labels, dtype=torch.long)
    data.y = y
    # print(data)

    return data

def prepare_GNN_data(data):
    labels = fetch_labels()

    # Split indices for training, validation, and testing
    train_idx, test_idx = train_test_split(range(len(labels)), test_size=0.3, stratify=labels)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, stratify=labels[test_idx])

    # Convert to tensors
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    # print(data)

    return data
