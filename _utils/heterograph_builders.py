# -*- coding: utf-8 -*-
"""
Created on 2023-11-01 (Wed) 17:11:07

heterograph builders

@author: I.Azuma
"""
# %%
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
from scipy import stats
from sklearn.decomposition import TruncatedSVD

from PIL import Image

import dgl
import torch

import sys
sys.path.append('/workspace/home/azuma/github/HeteroGraph_Pathology')
from _utils import graph_builders,cell_feature_extractor_legacy,heterograph_builders,visualizers

# %% tissue-cell heterogeneous graph

class HeteroGraphBuilders():
    def __init__(self):
        self.cell_graph_list = []
        
    def purified_cg_from_hovernet(self,
                                image_path = '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Images/train_6.png',
                                mat_path = '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/mat/train_6.mat',
                                json_path = '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/json/train_6.json',
                                true_path = '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Labels/train_6.mat',
                                neighbor_k=5,thresh=50,ignore_labels=[0]):
        """
        1. True labels are obtained by matching the dtected cell regions with the correct instamce map.
        2. If the labels assigned are background or cell types to be ignored, they are excluded from the cell graph.
        """
        # 0. image
        image = np.array(Image.open(image_path))
        # 1. instance map
        mat_info = sio.loadmat(mat_path)
        inst_map = mat_info['inst_map']
        # 2. load json file
        with open(json_path) as json_file:
            info = json.load(json_file)
        info = info['nuc']

        # 3. node feature
        # TODO: Improve to allow input of external information
        cfe = cell_feature_extractor_legacy.CellFeatureExtractor(mat_path=mat_path,json_path=json_path)
        cfe.load_data()
        node_feature = cfe.conduct()
        node_feature = node_feature[1::] # avoid background

        # true map
        true_map =  sio.loadmat(true_path)['type_map']

        # run
        error_counter = 0
        ignore_counter = 0
        remove_cell_idx = []
        new_instances = [0] # add background at first
        centroids = []
        update_label = 0
        type_list = []
        true_list = []
        update_info = []
        for inst_l in tqdm(range(1,inst_map.max()+1)):
            cent = info[str(inst_l)]['centroid']
            x = int(round(cent[0]))
            y = int(round(cent[1]))

            tmp_inst = np.where(inst_map==inst_l)
            inst_labels = inst_map[tmp_inst[0],tmp_inst[1]]
            inst_freq = int(stats.mode(inst_labels, axis=None).mode)
            # Detected cells recognized as background labels
            if inst_freq == 0:
                error_counter += 1
                remove_cell_idx.append(inst_l)
                new_instances.append(0)
            else:
                true_labels = true_map[tmp_inst[0],tmp_inst[1]]
                true_labels = [t for t in true_labels if t != 0] # remove background
                # All detected pixels are background labels
                if len(true_labels) == 0:
                    ignore_counter += 1
                    remove_cell_idx.append(inst_l)
                    new_instances.append(0)
                else:
                    # Most frequent labels other than background labels
                    true_freq = int(stats.mode(true_labels, axis=None).mode)
                    if true_freq in ignore_labels:
                        ignore_counter += 1
                        remove_cell_idx.append(inst_l)
                        new_instances.append(0)
                    else:
                        centroids.append([int(round(cent[0])),int(round(cent[1]))])
                        update_label += 1 # Shift the instance number
                        new_instances.append(update_label)
                        type_list.append(info[str(inst_l)]['type']) # original type
                        true_list.append(true_freq) # updated type (true label for training)
                        update_info.append(info[str(inst_l)])

        convert_dict = dict(zip([i for i in range(len(new_instances)+1)],new_instances))
        updated_info = dict(zip([str(i+1) for i in range(len(update_info))], update_info))

        # update instance map (time consuming)
        inst_df = pd.DataFrame(inst_map)
        fxn = lambda x : convert_dict.get(x)
        update_inst = inst_df.applymap(fxn)
        update_inst_map = np.array(update_inst)
        del inst_df,update_inst

        # update node feature
        update_node_feature = np.delete(node_feature, [r-1 for r in remove_cell_idx], 0)

        dat = graph_builders.CentroidsKNNGraphBuilder(k=neighbor_k, thresh=thresh, add_loc_feats=False)
        cell_graph = dat.process(instance_map=update_inst_map,features=update_node_feature,centroids=centroids)

        # checkpoint
        if np.max(update_inst_map) != cell_graph.num_nodes():
            raise ValueError('!! Something is wrong in creating cell graph !!')

        return cell_graph, update_inst_map, centroids, type_list, true_list, updated_info


    def cg_from_hovernet(self,
                        image_path = '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Test/Images/test_10.png',
                        mat_path = '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/mat/test_10.mat',
                        json_path = '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/json/test_10.json',neighbor_k=5,thresh=50):
        # 0. image
        image = np.array(Image.open(image_path))
        # 1. instance map
        inst_map = sio.loadmat(mat_path)['inst_map']
        # 2. node feature
        cfe = cell_feature_extractor_legacy.CellFeatureExtractor(mat_path=mat_path,json_path=json_path)
        cfe.load_data()
        node_feature = cfe.conduct()
        node_feature = node_feature[1::] # avoid background
        # 3. centroids
        with open(json_path) as json_file:
            info = json.load(json_file)
        info = info['nuc']
        centroids = np.empty((len(info), 2))
        for i,k in enumerate(info):
            cent = info[k]['centroid']
            centroids[i,0] = int(round(cent[0]))
            centroids[i,1] = int(round(cent[1]))
        # cell type label
        type_list = []
        for i,k in enumerate(info):
            type_list.append(info[k]['type'])

        dat = graph_builders.CentroidsKNNGraphBuilder(k=neighbor_k, thresh=thresh, add_loc_feats=False)
        cell_graph = dat.process(instance_map=inst_map,features=node_feature,centroids=centroids)

        return cell_graph, type_list


    def multiimage_tissue_cell_heterograph(self,image_path_list=[],mat_path_list=[],json_path_list=[],
                cell_feature_list=[],tissue_feature_list=[],superpixel_list=[],true_label_list=[],
                feat_svd=True,feature_dim=32,image_type=[0,0,0,1],tti_threshold=0.7,tti_k=5):
        """_summary_

        Args:
            image_path_list (_type_):
                image_path_list = [
                    '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Images/train_6.png',
                    '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Images/train_10.png',
                    '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Images/train_20.png',
                    '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Test/Images/test_10.png'
                ]
            mat_path_list (_type_):
                mat_path_list = [
                    '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/mat/train_6.mat',
                    '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/mat/train_10.mat',
                    '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/mat/train_20.mat',
                    '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/test/mat/test_10.mat'
                ]
            json_path_list (_type_):
                json_path_list = [
                    '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/json/train_6.json',
                    '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/json/train_10.json',
                    '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/json/train_20.json',
                    '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/test/json/test_10.json'
                ]
            tissue_feature_list (_type_):
                tissue_feature_list = [
                '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/train_6_tissue_features.pkl',
                '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/train_10_tissue_features.pkl',
                '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/train_20_tissue_features.pkl',
                '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/test_10_tissue_features.pkl'
            ]
            superpixel_list (_type_):
                superpixel_list = [
                '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/trian_6_superpixel.pkl',
                '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/trian_10_superpixel.pkl',
                '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/trian_20_superpixel.pkl',
                '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/test_10_superpixel.pkl',
            ]
            true_label_list (_type_):
                true_label_list = [
                '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Labels/train_6.mat',
                '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Labels/train_10.mat',
                '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Labels/train_20.mat',
                '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Test/Labels/test_10.mat',
            ]
            feature_dim (int, optional): _description_. Defaults to 32.

        """
        t_max = 0
        c_max = 0

        final_tissue_labels = []
        final_cell_labels = []
        final_cci_s = []
        final_cci_d = []
        final_estimated_type = [] # estimated type list
        final_true_type = []
        train_update_info = []
        for idx in range(len(image_type)):
            # load cell graph and estimated type list
            # graphs for train and valid
            if image_type[idx] == 0:
                cell_graph, update_inst_map, centroids, type_list, true_list, update_info = self.purified_cg_from_hovernet(image_path=image_path_list[idx],mat_path=mat_path_list[idx],json_path=json_path_list[idx],true_path=true_label_list[idx],neighbor_k=5,thresh=50,ignore_labels=[0])
                train_update_info.append(update_info)
            # graphs for test
            else:
                cell_graph, type_list = self.cg_from_hovernet(image_path=image_path_list[idx],mat_path=mat_path_list[idx],json_path=json_path_list[idx])

                # load instance and true map
                inst_map = sio.loadmat(mat_path_list[idx])['inst_map']
                true_map = sio.loadmat(true_label_list[idx])['type_map']
                with open(json_path_list[idx]) as json_file:
                    info = json.load(json_file)
                info = info['nuc']
                type_list, true_list = instance_true_assignment(inst_map,true_map,info,ignore_labels=[0])
            
            final_true_type.extend(true_list)
            self.cell_graph_list.append(cell_graph)

            # obtain initial cell and tissue feature
            if len(cell_feature_list) == 0:
                cell_feature = cell_graph.ndata['feat']
            else:
                cell_feature = pd.read_pickle(cell_feature_list[idx])

            tissue_feature = pd.read_pickle(tissue_feature_list[idx])
            superpixel = pd.read_pickle(superpixel_list[idx])
            final_estimated_type.extend(type_list)

            tissue_labels = []
            for centroids in cell_graph.ndata['centroid'].tolist():
                x = int(centroids[0])
                y = int(centroids[1])
                l = superpixel[y][x]
                tissue_labels.append(l)

            unique_idx = [k-1 for k in sorted(list(set(tissue_labels)))]
            target_feature = tissue_feature[[unique_idx]]
            tissue_labels = heterograph_builders.relabel(tissue_labels) # relabel

            # cell labels of cell-cell graph
            s = cell_graph.edges()[0].tolist()
            d = cell_graph.edges()[1].tolist()

            us = [es+c_max for es in s]
            ud = [ed+c_max for ed in d]
            cci_s = us+ud
            cci_d = ud+us

            final_cci_s.extend(cci_s)
            final_cci_d.extend(cci_d)

            # cell labels of tissue-cell graph
            tc_dest = [i for i in range(cell_graph.num_nodes())]  # tissue-cell destination (0,1,...,649)
            utc_dest = [tc+c_max for tc in tc_dest]
            final_cell_labels.extend(utc_dest)
            c_max = max(utc_dest)+1 # update c_max

            # update tissue labels
            ut = [tl+t_max for tl in tissue_labels]
            final_tissue_labels.extend(ut)
            t_max = max(ut)+1 # update t_max

            # concat each cell and tissue feature
            if idx == 0:
                merge_tissue_feature = target_feature
                merge_cell_feature = cell_feature
            else:
                merge_tissue_feature = torch.concat([merge_tissue_feature,target_feature])
                merge_cell_feature = torch.concat([merge_cell_feature,cell_feature])
            
            print('Cell Size: ',len(cell_graph.ndata['centroid']))
            print('Tissue Size: ',len(target_feature))
            print('True Label: ', set(true_list))

        # tissue-tissue interaction
        from sklearn.neighbors import kneighbors_graph
        # 1. kNN adjacency
        adj_t = kneighbors_graph(
                merge_tissue_feature,
                tti_k,
                mode="distance",
                include_self=False,
                metric="euclidean").toarray()
        adj_t = torch.tensor(np.array(adj_t))
        # 2. correlation based (legacy)
        """
        cor_adj = pd.DataFrame(merge_tissue_feature).T.corr()
        threshold = tti_threshold
        fxn = lambda x : x if (threshold < x)&(x<1) else 0
        cor_adj = cor_adj.applymap(fxn) # update
        adj_t = torch.tensor(np.array(cor_adj))
        """
        self.adj_t = adj_t
        edge_index = adj_t.nonzero().t().contiguous()
        ts = edge_index[0].tolist()
        td = edge_index[1].tolist()

        # process feature
        if feat_svd:
            svd = TruncatedSVD(n_components=feature_dim, random_state=1) # tissue feature
            merge_tissue_feature = svd.fit_transform(merge_tissue_feature)
            svd = TruncatedSVD(n_components=feature_dim, random_state=1) # cell feature
            merge_cell_feature = svd.fit_transform(merge_cell_feature)

        # construct graph
        graph_data = {}
        graph_data[('tissue','tissue2cell','cell')] = (final_tissue_labels, final_cell_labels)
        graph_data[('cell','cell2tissue','tissue')] = (final_cell_labels, final_tissue_labels)
        #graph_data[('cell','cci','cell')] = (final_cci_s, final_cci_d)
        graph_data[('tissue','tti','tissue')] = (ts+td, td+ts)
        graph = dgl.heterograph(graph_data)
        edges = ['tissue2cell','cell2tissue','cci','tti']

        graph.nodes['tissue'].data['id'] = torch.ones(graph.num_nodes('tissue')).long()
        graph.nodes['cell'].data['id'] = torch.arange(graph.num_nodes('cell')).long()
        graph.nodes['tissue'].data['feat'] = torch.tensor(merge_tissue_feature)
        graph.nodes['cell'].data['feat'] = torch.tensor(merge_cell_feature)
        graph.edges['cell2tissue'].data['weight'] = torch.ones(graph['cell2tissue'].num_edges())
        graph.edges['tissue2cell'].data['weight'] = torch.ones(graph['tissue2cell'].num_edges())
        #graph.edges['cci'].data['weight'] = torch.ones(graph['cci'].num_edges())
        graph.edges['tti'].data['weight'] = torch.ones(graph['tti'].num_edges()) # torch.cat((adj_t[ts,td],adj_t[ts,td]))

        return graph, edges, final_estimated_type, relabel(final_true_type), train_update_info

def relabel(label_list=[3,10,21,5]):
    """_summary_

    Args:
        label_list (list, optional): _description_. Defaults to [3,10,21,5].

    Returns:
        list: [0,2,3,1]
    """
    unique_s = sorted(list(set(label_list)))
    relabel_dic = dict(zip(unique_s,[i for i in range(len(unique_s))]))
    relabel_l = [relabel_dic.get(k) for k in label_list]
    return relabel_l

def instance_true_assignment(inst_map,true_map,info,ignore_labels=[0]):
    """_summary_

    Args:
        inst_map (np.array): Detected instance map.
        true_map (np.array): True instance map.
        info (dict): Information derived from json file.
        ignore_labels (list, optional): _description_. Defaults to [0].

    Returns:
        type_list: Detected raw cell type list.
        true_list: Annotated true cell type list. Use for calc train loss.
    """
    centroids = []
    type_list = []
    true_list = []
    for inst_l in tqdm(range(1,inst_map.max()+1)):
        cent = info[str(inst_l)]['centroid']
        x = int(round(cent[0]))
        y = int(round(cent[1]))

        tmp_inst = np.where(inst_map==inst_l) # (x_array, y_array)
        inst_labels = inst_map[tmp_inst[0],tmp_inst[1]]
        inst_freq = int(stats.mode(inst_labels, axis=None).mode)

        true_labels = true_map[tmp_inst[0],tmp_inst[1]]
        true_freq = int(stats.mode(true_labels, axis=None).mode) # remove background

        centroids.append([x, y])
        type_list.append(info[str(inst_l)]['type'])
        true_list.append(true_freq)

    return type_list, true_list
