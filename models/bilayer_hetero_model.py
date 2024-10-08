# -*- coding: utf-8 -*-
"""
Created on 2023-11-14 (Tue) 19:39:18

Heterogeneous Graph Neural Network Model for tissue-cell interactinos in pathology.
- Tissue-Tissue Graph
- Cell-Cell Graph
- Tissue-Cell Graph

@author: I.Azuma
"""
# %%
import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD

# %%
class HeteroNet(nn.Module):

    def __init__(self, args, edges=[]):
        super().__init__()
        self.args = args
        self.edges = edges
        self.nrc = args.no_readout_concatenate
        self.inemb = args.initial_embedding

        hid_feats = args.hidden_size
        out_feats = args.output_size
        FEATURE_SIZE = args.cell_size

        self.embed_tissue = nn.Embedding(2, hid_feats)
        self.embed_feat = nn.Embedding(FEATURE_SIZE, hid_feats)

        self.lin1 = nn.Linear(2048, hid_feats)
        self.lin2 = nn.Linear(2048, hid_feats)

        self.input_linears = nn.ModuleList()
        self.input_acts = nn.ModuleList()
        self.input_norm = nn.ModuleList()
        for i in range((args.embedding_layers - 1) * 2):
            self.input_linears.append(nn.Linear(hid_feats, hid_feats))

        for i in range((args.embedding_layers - 1) * 2):
            self.input_acts.append(nn.GELU())
        for i in range((args.embedding_layers - 1) * 2):
            self.input_norm.append(nn.GroupNorm(4, hid_feats))

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            dglnn.HeteroGraphConv(
                dict(
                    zip(self.edges, [
                        dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, # FIXME
                        aggregator_type=args.agg_function, norm=None) for i in range(len(self.edges))
                    ])), aggregate='stack'))
        for i in range(args.conv_layers - 1):
            self.conv_layers.append(
                dglnn.HeteroGraphConv(
                    dict(
                        zip(self.edges, [
                            dglnn.SAGEConv(in_feats=hid_feats * 2, out_feats=hid_feats,
                                            aggregator_type=args.agg_function, norm=None)
                                            
                            for i in range(len(self.edges))
                        ])), aggregate='stack'))


        self.conv_acts = nn.ModuleList()
        self.conv_norm = nn.ModuleList()
        for i in range(args.conv_layers * 2):
            self.conv_acts.append(nn.GELU())

        for i in range(args.conv_layers * len(self.edges)):
            self.conv_norm.append(nn.GroupNorm(4, hid_feats))

        self.readout_linears = nn.ModuleList()
        self.readout_acts = nn.ModuleList()

        # readout concat
        if self.nrc:
            for i in range(args.readout_layers - 1):
                self.readout_linears.append(nn.Linear(hid_feats, hid_feats))
            self.readout_linears.append(nn.Linear(hid_feats, out_feats))
        else:
            for i in range(args.readout_layers - 1):
                self.readout_linears.append(nn.Linear(hid_feats * args.conv_layers, hid_feats * args.conv_layers))
            self.readout_linears.append(nn.Linear(hid_feats * args.conv_layers, out_feats))

        for i in range(args.readout_layers - 1):
            self.readout_acts.append(nn.GELU())

        if args.hetero_aggregation == 'alpha' and args.cell_weight < 0:
            self.aph = nn.Parameter(torch.zeros(2))

    def attention_agg(self, layer, h0, h, weight_v):
        # FIXME: not using h0
        # h: h^{l-1}, dimension: (batch, hidden)
        # feats: result from two conv(cell conv and pathway conv), stacked together; dimension: (batch, 2, hidden)
        args = self.args
        if h.shape[1] == 1:
            # when no hetero reletionships (e.g. edges = ['tissue2cell', 'cell2tissue'])
            return self.conv_norm[layer * len(self.edges) + 1](h.squeeze(1))
        elif args.hetero_aggregation == 'sum':
            return h[:, 0, :] + h[:, 1, :]
        else:
            # FIXME: Depends on the edges order.
            # 1. inter relationship (e.g. tissue to cell)
            h1 = h[:, 0, :]
            # 2. inner relationship (e.g. various types of cell to cell)
            for pi in range(1,h.shape[1]):
                if pi == 1:
                    h2 = h[:, pi, :]
                else:
                    h2 += h[:, pi, :]

            if args.subpath_activation:
                h1 = F.leaky_relu(h1)
                h2 = F.leaky_relu(h2)

            h1 = self.conv_norm[layer * len(self.edges) + 1](h1)
            h2 = self.conv_norm[layer * len(self.edges) + 2](h2)

        # aggregation
        if weight_v < 0:
            weight = torch.softmax(self.aph, -1)
            return weight[0] * h1 + weight[1] * h2
        else:
            return (1 - weight_v) * h1 + weight_v * h2

    def conv(self, graph, layer, h, hist):
        args = self.args
        h0 = hist[-1] # latest information
        h = self.conv_layers[layer](graph, h, mod_kwargs=dict(
            zip(self.edges, [{
                'edge_weight':
                F.dropout(graph.edges[self.edges[i]].data['weight'], p=args.edge_dropout, training=self.training)
            } for i in range(len(self.edges))])))
            
        if args.model_dropout > 0:
            h = {
                'cell':
                F.dropout(self.conv_acts[layer * 2](self.attention_agg(layer, h0['cell'], h['cell'], args.cell_weight)),
                          p=args.model_dropout, training=self.training),
                'tissue':
                F.dropout(self.conv_acts[layer * 2 + 1](self.attention_agg(layer, h0['tissue'], h['tissue'], args.tissue_weight)),
                          p=args.model_dropout, training=self.training)
            }
        else:
            h = {
                'cell': self.conv_acts[layer * 2](self.attention_agg(layer, h0['cell'], h['cell'], args.cell_weight)),
                'tissue': self.conv_acts[layer * 2 + 1](self.attention_agg(layer, h0['tissue'], h['tissue'], args.tissue_weight))
            }

        return h

    def calculate_initial_embedding(self, graph):
        args = self.args

        input1 = F.leaky_relu(self.embed_feat(graph.srcdata['id']['cell']))
        input2 = F.leaky_relu(self.embed_tissue(graph.srcdata['id']['tissue']))

        hcell = input1
        htissue = input2
        for i in range(args.embedding_layers - 1, (args.embedding_layers - 1) * 2):
            hcell = self.input_linears[i](hcell)
            hcell = self.input_acts[i](hcell)
            if args.normalization != 'none':
                hcell = self.input_norm[i](hcell)
            if args.model_dropout > 0:
                hcell = F.dropout(hcell, p=args.model_dropout, training=self.training)

        for i in range(args.embedding_layers - 1):
            htissue = self.input_linears[i](htissue)
            htissue = self.input_acts[i](htissue)
            if args.normalization != 'none':
                htissue = self.input_norm[i](htissue)
            if args.model_dropout > 0:
                htissue = F.dropout(htissue, p=args.model_dropout, training=self.training)

        return hcell, htissue
    
    def process_initial_feature(self, graph):
        args = self.args

        cell_feature = graph.srcdata['feat']['cell']
        tissue_feature = graph.srcdata['feat']['tissue']

        hcell = self.lin1(cell_feature)
        htissue = self.lin2(tissue_feature)

        return hcell, htissue
    
    def process_initial_feature_with_sampling(self, blocks):
        args = self.args

        # Extract cell and tissue features from each block
        hcell_list = []
        htissue_list = []
        
        for block in blocks:
            # Obtaine the source node features
            cell_feature = block.srcdata['feat']['cell']
            tissue_feature = block.srcdata['feat']['tissue']

            # Linear transformation
            hcell = self.lin1(cell_feature)
            htissue = self.lin2(tissue_feature)

            hcell_list.append(hcell)
            htissue_list.append(htissue)

        return hcell_list, htissue_list

    def propagate(self, graph):
        args = self.args
        if self.inemb:
            hcell, htissue = self.calculate_initial_embedding(graph)
        else:
            hcell, htissue = self.process_initial_feature(graph)

        h = {'cell': hcell, 'tissue': htissue}
        hist = [h]

        for i in range(args.conv_layers):
            if i == 0 or args.residual == 'none':
                pass
            elif args.residual == 'res_add':
                if args.initial_residual:
                    h = {'cell': h['cell'] + hist[0]['cell'], 'tissue': h['tissue'] + hist[0]['tissue']}

                else:
                    h = {'cell': h['cell'] + hist[-2]['cell'], 'tissue': h['tissue'] + hist[-2]['tissue']}

            elif args.residual == 'res_cat':
                if args.initial_residual:
                    h = {
                        'cell': torch.cat([h['cell'], hist[0]['cell']], 1),
                        'tissue': torch.cat([h['tissue'], hist[0]['tissue']], 1)
                    }
                else:
                    h = {
                        'cell': torch.cat([h['cell'], hist[-2]['cell']], 1),
                        'tissue': torch.cat([h['tissue'], hist[-2]['tissue']], 1)
                    }

            h = self.conv(graph, i, h, hist)
            hist.append(h)

        return hist
    
    def propagate_with_sampling(self, blocks:list):
        args = self.args
        emb_flag = 0
        # # Initialize features for the first layer
        if self.inemb:
            emb_flag += 1
            hcell, htissue = self.calculate_initial_embedding(blocks[0])
        else:
            hcell, htissue = self.process_initial_feature(blocks[0])

        h = {'cell': hcell, 'tissue': htissue}

        for i in range(args.conv_layers):
            if i > 0:
                if emb_flag > 0:
                    hcell0, htissue0 = self.calculate_initial_embedding(blocks[i])
                else:
                    hcell0, htissue0 = self.process_initial_feature(blocks[i])    
                h = {'cell': torch.cat([h['cell'], hcell0], 1), 'tissue': torch.cat([h['tissue'], htissue0], 1)}
            

            hist = [h]
            h = self.conv(blocks[i], i, h, hist)
        
        hist = [h] * (args.conv_layers + 1)
        return hist


    def forward(self, graph):
        args = self.args

        # if the graph type is list >> block sampling
        if isinstance(graph, list):
            hist = self.propagate_with_sampling(graph)
        else:
            hist = self.propagate(graph)

        self.hist = hist

        # Concat the number of conv_layers other than the initial value.
        if not self.nrc:
            h = torch.cat([i['cell'] for i in hist[1:]], 1)
            # hist[0] contains initial_embedding
        else:
            h = hist[-1]['cell']

        for i in range(args.readout_layers - 1):
            h = self.readout_linears[i](h)
            h = F.dropout(self.readout_acts[i](h), p=args.model_dropout, training=self.training)
        h = self.readout_linears[-1](h)

        if args.output_relu == 'relu':
            return F.relu(h)
        elif args.output_relu == 'leaky_relu':
            return F.leaky_relu(h)

        #return h
        return F.log_softmax(h, dim=1)
