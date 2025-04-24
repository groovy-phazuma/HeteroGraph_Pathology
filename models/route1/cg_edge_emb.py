# -*- coding: utf-8 -*-
"""
Created on 2025-03-22 (Sat) 10:57:22

@author: I.Azuma
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

"""
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
"""
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x) 
        return x

def positional_encoding(bbox_i, bbox_j):
    """
    Calculate positional encoding from the bounding boxes of two cells.
    
    Parameters:
        bbox_i (torch.Tensor): bbox of cell_i, shape = (2, 2)
            e.g. torch.tensor([[993, 257], [1000, 273]])
        bbox_j (torch.Tensor): bbox of cell_j, shape = (2, 2)
            e.g. torch.tensor([[800, 500], [820, 520]])
    
    Returns:
        torch.Tensor: (dy, dx, log_h_ratio, log_w_ratio)
    """
    ri, ci = bbox_i[0, 0], bbox_i[0, 1]
    rj, cj = bbox_j[0, 0], bbox_j[0, 1]

    hi, wi = bbox_i[1, 0] - bbox_i[0, 0], bbox_i[1, 1] - bbox_i[0, 1]
    hj, wj = bbox_j[1, 0] - bbox_j[0, 0], bbox_j[1, 1] - bbox_j[0, 1]

    dy = 2 * (ri - rj) / (hi + hj)
    dx = 2 * (ci - cj) / (wi + wj)

    height_ratio = torch.log(hi / hj)
    width_ratio = torch.log(wi / wj)

    return torch.stack([dy, dx, height_ratio, width_ratio], dim=-1)

class MessagePassingNetwork(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, edge_index):
        super().__init__(aggr='mean')  # Mean aggregation as described in the paper
        
        self.Me = MLP(2 * node_dim + edge_dim, hidden_dim, edge_dim)  # Edge update function
        self.Ma = MLP(node_dim + edge_dim, hidden_dim, node_dim)  # Node update function

        self.edge_index = edge_index
        

    def forward(self, node_feats, edge_feats, t=3):
        for _ in range(t):
            edge_feats = self.update_edges(node_feats, edge_feats)
            node_feats = self.update_nodes(node_feats, edge_feats)
        return node_feats, edge_feats

    def update_edges(self, node_feats, edge_feats):
        row, col = self.edge_index
        edge_inputs = torch.cat([node_feats[row], node_feats[col], edge_feats], dim=1)
        return self.Me(edge_inputs)

    def update_nodes(self, node_feats, edge_feats):
        row, col = self.edge_index
        aggregated_messages = torch.zeros_like(node_feats)
        aggregated_messages.index_add_(0, row, self.Ma(torch.cat([node_feats[row], edge_feats], dim=1)))
        return aggregated_messages


class GNNClassifierStatic(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_classes, edge_index, g, bbox_info, features):
        super().__init__()
        self.edge_index = edge_index
        self.features = features

        # Initialize edge features
        centroids = g.ndata['centroid']
        edge_pos_encodings = []
        for i, (u, v) in enumerate(edge_index.T):
            bbox1 = bbox_info[u] 
            bbox2 = bbox_info[v]
            pe = positional_encoding(bbox1, bbox2)  # 1. calculate edge positional encodings
            node_diff = torch.norm(features[u] - features[v], p=2).unsqueeze(0)  # 2. node feature difference
            euclidean_dist = torch.norm(centroids[u] - centroids[v], p=2).unsqueeze(0)  # 3. euclidean distance
            edge_pos_encodings.append(torch.cat([pe, node_diff, euclidean_dist], dim=-1))
            
        self.edge_feats = torch.stack(edge_pos_encodings, dim=0)

        self.gnn = MessagePassingNetwork(node_dim, edge_dim, hidden_dim, edge_index)
        self.node_classifier = MLP(node_dim, hidden_dim, num_classes)
        self.edge_classifier = MLP(edge_dim, hidden_dim, 2)

        # Label for node classification
        self.node_labels = g.ndata['label']
        self.edge_labels = g.edata['label']  # Assuming edge labels are stored in the graph

    def forward(self, edge_index, t=3):
        node_features = self.features
        edge_features = self.edge_feats
        node_features, edge_features = self.gnn(node_features, edge_features, t)
        print(edge_features)

        node_preds = self.node_classifier(node_features)
        edge_preds = self.edge_classifier(edge_features)

        return node_preds, edge_preds


class GNNClassifierDynamic(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_classes, edge_index, g, bbox_info):
        super().__init__()
        self.edge_index = edge_index
        self.bbox_info = bbox_info
        self.g = g

        self.gnn = MessagePassingNetwork(node_dim, edge_dim, hidden_dim, edge_index)
        self.node_classifier = MLP(node_dim, hidden_dim, num_classes)
        self.edge_classifier = MLP(edge_dim, hidden_dim, 2)

        self.node_labels = g.ndata['label']
        self.edge_labels = g.edata['label']

    def compute_edge_features(self, features):
        centroids = self.g.ndata['centroid']
        edge_pos_encodings = []
        for u, v in self.edge_index.T:
            bbox1 = self.bbox_info[u]
            bbox2 = self.bbox_info[v]
            pe = positional_encoding(bbox1, bbox2)
            node_diff = torch.norm(features[u] - features[v], p=2).unsqueeze(0)
            euclidean_dist = torch.norm(centroids[u] - centroids[v], p=2).unsqueeze(0)
            edge_pos_encodings.append(torch.cat([pe, node_diff, euclidean_dist], dim=-1))
        edge_feats = torch.stack(edge_pos_encodings, dim=0)
        return edge_feats

    def forward(self, features, edge_index=None, t=3):
        if edge_index is None:
            edge_index = self.edge_index
        
        edge_features = self.compute_edge_features(features)
        node_features, edge_features = self.gnn(features, edge_features, t)
        node_preds = self.node_classifier(node_features)
        edge_preds = self.edge_classifier(edge_features)
        
        return node_preds, edge_preds
