# -*- coding: utf-8 -*-
"""
Created on 2024-01-22 (Mon) 20:11:36

@author: I.Azuma
"""
# %%
import importlib

import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor

from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
print(BASE_DIR)

from preprocessing import image_utils

class MyCell2Patch(Dataset):
    """
    Extract patches that place each cell in the center.
    """
    def __init__(self, image_path:str='', json_path:str='', image=None, nuc_info=None, ext_method="centroid") -> None:
        super().__init__()
        if image is None:
            self.image = Image.open(image_path).convert("RGB") # Load image
        else:
            self.image = image

        if nuc_info is None:
            with open(json_path) as json_file: # Load nuclei information (HoVerNet)
                data = json.load(json_file)
            nuc_info = data['nuc']
        self.ext_method = ext_method

        self.all_patches, self.coords = image_utils.cell_patch_extractor(self.image, nuc_info, ext_method=self.ext_method,patch_size=72)
        self.nr_patches = len(self.all_patches)

        self.dataset_transform = transforms.Compose(
            [
                #transforms.Resize((224, 224)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        )
    
    def __getitem__(self, index: int):
        patch = self.all_patches[index]
        transformed_patch = self.dataset_transform(patch)
        if self.ext_method == "bbox":
            coord = self.coords[index]
        else:
            coord = (0,0)

        return coord, transformed_patch

    def __len__(self) -> int:
        return self.nr_patches

class MyRegion2Patch(Dataset):
    """
    Extract patches that best reflect superpixel instance information.
    """
    def __init__(self, image_path:str, superpixels:np.array, ext_method="bbox", stepsize=10, windowsize=(144,144), mask_others=False) -> None:
        super().__init__()
        self.image = Image.open(image_path).convert("RGB") # Load image
        self.superpixels = superpixels
        self.stepsize=stepsize
        self.windowsize=windowsize

        self.all_patches, self.upperlefts = image_utils.tissue_patch_extractor(self.image, self.superpixels, ext_method, stepsize=self.stepsize, windowsize=self.windowsize, mask_others=mask_others)
        self.nr_patches = len(self.all_patches)

        self.dataset_transform = transforms.Compose(
            [
                #transforms.Resize((224, 224)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        )

    def __getitem__(self, index: int):
        patch = self.all_patches[index]
        transformed_patch = self.dataset_transform(patch)
        upperleft = self.upperlefts[index]

        return upperleft, transformed_patch

    def __len__(self) -> int:
        return self.nr_patches

class LoadModel():
    def __init__(self,architecture='resnet50'):
        self.architecture = architecture
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")
    
    def dynamic_import_from(self, source_file: str, class_name: str) -> Any:
        """Do a from source_file import class_name dynamically

        Args:
            source_file (str): Where to import from
            class_name (str): What to import

        Returns:
            Any: The class to be imported
        """
        module = importlib.import_module(source_file)
        return getattr(module, class_name)
    
    def _remove_layers(self, model: nn.Module, extraction_layer: Optional[str] = None) -> nn.Module:
        """
        Returns the model without the unused layers to get embeddings.

        Args:
            model (nn.Module): Classification model.

        Returns:
            nn.Module: Embedding model.
        """
        if hasattr(model, "model"):
            model = model.model
            if extraction_layer is not None:
                model = _remove_modules(model, extraction_layer)
        if isinstance(model, torchvision.models.resnet.ResNet):
            if extraction_layer is None:
                # remove classifier
                model.fc = nn.Sequential()
            else:
                # remove all layers after the extraction layer
                model = _remove_modules(model, extraction_layer)
        else:
            # remove classifier
            model.classifier = nn.Sequential()
            if extraction_layer is not None:
                # remove average pooling layer if necessary
                if hasattr(model, 'avgpool'):
                    model.avgpool = nn.Sequential()
                # remove all layers in the feature extractor after the extraction layer
                model.features = _remove_modules(model.features, extraction_layer)
        return model
    
    def load_model(self,remove_layer=True):
        model_class = self.dynamic_import_from("torchvision.models", self.architecture)
        model = model_class(pretrained=True)
        model = model.to(self.device)
        if remove_layer:
            model = self._remove_layers(model, extraction_layer=None)
        model.eval()
        self.model = model

def _remove_modules(model: nn.Module, last_layer: str) -> nn.Module:
    """
    Remove all modules in the model that come after a given layer.

    Args:
        model (nn.Module): A PyTorch model.
        last_layer (str): Last layer to keep in the model.

    Returns:
        nn.Module: Model without pruned modules.
    """
    modules = [n for n, _ in model.named_children()]
    modules_to_remove = modules[modules.index(last_layer)+1:]
    for mod in modules_to_remove:
        setattr(model, mod, nn.Sequential())
    return model

def flex_feat_extractor(image_loader, model, target_layers=['layer1.2.relu_2','layer2.3.relu_2','layer3.5.relu_2','layer4.2.relu_2']):
    feat_extractor = create_feature_extractor(model, target_layers)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    features = torch.Tensor()
    for coords, image_batch in tqdm(image_loader):
        image_batch = image_batch.to(device)

        with torch.no_grad():
            
            feat_dict = feat_extractor(image_batch)
            #emb = model(image_batch).cpu().squeeze() # Simple
            merged_emb = concat_avgpool(feat_dict)

        features = torch.cat((features, merged_emb))
    
    for i, l in enumerate(feat_dict):
        print('Name:{}, Size:{}'.format(target_layers[i],feat_dict.get(l).shape[1]))
    
    return features

def concat_avgpool(feat_dict):
    adaptiveavgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    merged_emb = torch.Tensor()
    for i, layer in enumerate(feat_dict):
        tmp_map = feat_dict.get(layer)
        output = adaptiveavgpool(tmp_map)

        emb = output.view(output.shape[0],-1).cpu()
        merged_emb = torch.concat([merged_emb, emb],axis=1)
    
    return merged_emb
