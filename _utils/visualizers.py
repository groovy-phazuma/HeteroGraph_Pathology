# -*- coding: utf-8 -*-
"""
Created on 2023-11-02 (Thu) 17:43:56

reference:
https://github.com/vqdang/hover_net/tree/67e2ce5e3f1a64a2ece77ad1c24233653a9e0901/misc


Note:
Use histocartography if you want to draw cell graphs.

```
import sys
sys.path.append(BASE_DIR+'/github/histocartography')
from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization

visualizer = OverlayGraphVisualization(
    instance_visualizer=InstanceImageVisualization(
    instance_style="filled+outline"
    )
)
viz_cg = visualizer.process(
    canvas=image,
    graph=cell_graph,
    instance_map=update_inst_map
)
```
@author: I.Azuma
"""
import cv2
import json 
import random
import colorsys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def random_colors(N, bright=True):
    """Generate random colors.
    
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def overlay_viz(image,inst_dict,type_colour=None):
    if type_colour is None:
        # https://github.com/vqdang/hover_net/blob/master/type_info.json
        type_colour = {
                0: ("nolabe", (0, 0, 0)),  # no label
                1: ("neopla", (255, 0, 0)),  # neoplastic
                2: ("inflam", (0, 255, 0)),  # inflamm
                3: ("connec", (0, 0, 255)),  # connective
                4: ("necros", (255, 255, 0)),  # dead
                5: ("no-neo", (255, 165, 0)),  # non-neoplastic epithelial
            }

    overlay = np.copy((image))
    inst_rng_colors = random_colors(len(inst_dict))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        final_contour = tuple([np.array([t]) for t in inst_contour]) # sensitive format
        """ like this format.
        (array([[135, 986]]),
        array([[134, 987]]),
        array([[140, 988]]),
        array([[139, 988]]),
        array([[138, 987]]),
        array([[136, 987]]))
        """
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            inst_colour = (inst_rng_colors[idx]).tolist()
        output = cv2.drawContours(overlay, final_contour, -1, inst_colour, 3)
    return output
    #cv2.imwrite('./output.png', output)

def before_after(pred, s=0, e=650,
                 image_path = '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Images/train_6.png',
                 true_overlay_path = '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Overlay/train_6.png',
                 json_path = '/train/json/train_6.json',
                 inst_dict = None):
    """_summary_

    Parameters
    ----------
    pred : _type_
        _description_
    s : int, optional
        _description_, by default 0
    e : int, optional
        _description_, by default 650
    image_path : str, optional
        _description_, by default '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Images/train_6.png'
    true_overlay_path : str, optional
        _description_, by default '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Overlay/train_6.png'
    json_path : str, optional
        _description_, by default '/train/json/train_6.json'
    inst_dict : _type_, optional
        _description_, by default None
    """
    image = np.array(Image.open(image_path))
    
    # PanNuke pretrained mode
    train_colour = {
    0 : ["nolabe", [0  ,   0,   0]], 
    1 : ["neopla", [255,   0,   0]], 
    2 : ["inflam", [255,  11, 255]], 
    3 : ["connec", [0  ,   0, 255]], 
    4 : ["necros", [255, 255,   0]], 
    5 : ["no-neo", [0  , 255,   0]] 
    }
    """ used when evaluationg test_10.png only.
    test_colour = {
        0 : ["nolabe", [0  ,   0,   0]], 
        1 : ["necros", [255, 255,   0]], 
        2 : ["inflam", [255,  11, 255]], 
        3 : ["no-neo", [0  , 255,   0]], 
        4 : ["neopla", [255,   0,   0]], 
        5 : ["connec", [0  ,   0, 255]],
        6 : ["muscle", [255, 255, 255]]
    }
    """

    # see https://arxiv.org/pdf/1812.06499
    test_colour = {
        0 : ["background",                      [0  ,   0,   0]],  # background
        1 : ["miscellaneous",                   [255,  255,  0]],  # yellow
        2 : ["inflam",                          [255,  11, 255]],  # pink
        3 : ["healthy_epithelical",             [0,   255,   0]],  # green
        4 : ["dysplastic_malignant_epithelial", [255, 0,     0]],  # red
        5 : ["fibroblast",                      [0,   0,   255]],  # blue
        6 : ["muscle",                          [0  , 255, 255]],  # cyan
        7 : ["endothelial",                     [255, 140,   0]],  # orange
    }

    if inst_dict is None:
        with open(json_path) as json_file:
            inst_dict = json.load(json_file)['nuc']
    
    # before
    overlay = overlay_viz(image=image,inst_dict=inst_dict,type_colour=train_colour)

    # after (update inst_dict)
    # _, pred = best_logits.max(dim=1)
    test_pred = pred[s:e] # slice
    counter = 0
    for i,k in enumerate(test_pred):
        old_label = inst_dict.get(str(i+1))['type']
        new_label = int(test_pred[i])
        if old_label != new_label:
            counter += 1
        inst_dict.get(str(i+1))['type'] = new_label
    overlay2 = overlay_viz(image=image,inst_dict=inst_dict,type_colour=test_colour)
    true_image = np.array(Image.open(true_overlay_path))

    fig, ax = plt.subplots(figsize=(15,5))
    ax.axis("off")
    
    plt1 = fig.add_subplot(1,3,1)
    plt1.imshow(true_image)
    plt1.set_title('True')

    plt2 = fig.add_subplot(1,3,2)
    plt2.imshow(overlay)
    plt2.set_title('HoverNet')

    plt3 = fig.add_subplot(1,3,3)
    plt3.imshow(overlay2)
    plt3.set_title('HoverNet + Proposed')

    plt.show()
