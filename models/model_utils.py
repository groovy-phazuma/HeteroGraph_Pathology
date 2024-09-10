# -*- coding: utf-8 -*-
"""
Created on 2024-09-10 (Tue) 20:48:19

utils for training and evaluation

@author: I.Azuma
"""
# %%
import torch
import torch.nn.functional as F

# %%
def predict(model, graph, idx=None, **kwargs):
        """Predict function to get latent representation of data.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Tissue-cell graph contructed from the dataset.
        idx : Iterable[int] optional
            Tissue indices for prediction, by default to be HeteroGraph_Pathology/models/bilayer_hetero_model.pyNone, where all the cells to be predicted.
        device : str optional
            Well to perform predicting, by default to be 'gpu'.

        Returns
        -------
        pred : torch.Tensor
            Predicted target modality features.

        """

        model.eval()
        with torch.no_grad():
            if idx is None:
                pred = model.forward(graph)
            else:
                pred = model.forward(graph)[idx]
        return pred

def score(model, g, idx, labels, **kwargs):
    """Score function to get score of prediction.

    Parameters
    ----------
    g : dgl.DGLGraph
        Tissue-cell graph contructed from the dataset.
    idx : Iterable[int] optional
        Index of testing cells for scoring.
    labels : torch.Tensor
        Ground truth label of cells, a.k.s target modality features.
    device : str optional
        Well to perform predicting, by default to be 'gpu'.

    Returns
    -------
    loss : float
        RMSE loss of predicted output modality features.

    """

    model.eval()
    with torch.no_grad():
        logits = predict(model, g, idx,**kwargs)
        loss = F.nll_loss(logits, labels)
    
    return loss.item()
