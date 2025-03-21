# -*- coding: utf-8 -*-
"""
Created on 2025-03-21 (Fri) 13:24:53

Node (cell) classification with GAT

@author: I.Azuma
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_heads=4, dropout=0.2):
        super(GATClassifier, self).__init__()
        self.gat1 = dglnn.GATConv(in_dim, hidden_dim, num_heads, feat_drop=dropout, attn_drop=dropout, activation=F.relu)
        self.gat2 = dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop=dropout, attn_drop=dropout, activation=F.relu)
        self.classifier = nn.Linear(hidden_dim * num_heads, num_classes)

    def forward(self, g, features):
        h = self.gat1(g, features)  # GATの1層目
        h = h.view(h.shape[0], -1)  # ヘッドを結合
        h = self.gat2(g, h)  # GATの2層目
        h = h.view(h.shape[0], -1)  # ヘッドを結合
        logits = self.classifier(h)  # 最終分類層
        return logits

class GATClassifierWithMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_heads=4, dropout=0.2):
        super(GATClassifierWithMLP, self).__init__()
        # GATの1層目
        self.gat1 = dglnn.GATConv(in_dim, hidden_dim, num_heads, feat_drop=dropout, attn_drop=dropout, activation=F.relu)
        # GATの2層目
        self.gat2 = dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop=dropout, attn_drop=dropout, activation=F.relu)
        
        # MLP層
        self.fc1 = nn.Linear(hidden_dim * num_heads, 512)  # 隠れ層1
        self.fc2 = nn.Linear(512, 256)  # 隠れ層2
        self.classifier = nn.Linear(256, num_classes)  # 出力層

    def forward(self, g, features):
        # GATの1層目
        h = self.gat1(g, features)
        h = h.view(h.shape[0], -1)  # ヘッドを結合

        # GATの2層目
        h = self.gat2(g, h)
        h = h.view(h.shape[0], -1)  # ヘッドを結合

        # MLPを通す
        h = F.relu(self.fc1(h))  # 隠れ層1
        h = F.relu(self.fc2(h))  # 隠れ層2
        logits = self.classifier(h)  # 最終分類層
        return logits


