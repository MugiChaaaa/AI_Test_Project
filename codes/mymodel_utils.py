### Import Torch Libraries
import torch
import torchvision.models as tmodels

### Import Other Libraries
import math
import numpy as np
from typing import Sequence


def get_cnn_channels(initial_ch: int = 32, length: int = 2, multiplier: float = 2.0) -> tuple:
    """
    Get the number of channels for the CNN model.
    :param initial_ch: Initial number of channels. Default is 32.
    :param length: Number of layers. Default is 2.
    :param multiplier: Multiplier for the number of channels. Default is 2.0.
    :return: _cnn_ch: tuple of number of channels for the CNN model.
    """
    _cnn_ch = (initial_ch, )
    for i in range(length - 1):
        _cnn_ch = _cnn_ch + (int(multiplier * _cnn_ch[-1]), )
    return _cnn_ch


def get_cnn_feature_size(initial_features: int = 2048, length: int = 2, output_features: int = 10) -> tuple:
    """
    Get the feature size for the CNN model.
    :param initial_features: Initial number of features. Default is 2048.
    :param length: Number of layers. Default is 2.
    :param output_features: Number of output features. Default is 10.
    :return: _feature_size: tuple of feature size for the CNN model. 'output_features' is not included.
    """
    _feature_size = (initial_features, )
    y1 = math.log2(initial_features)
    y2 = math.log2(output_features)
    for i in range(length - 1):
        m = (length - i - 1) / length
        n = 1 - m
        exp_value = int(y1 * m + y2 * n)
        _feature_size = _feature_size + (int(pow(2, exp_value)), )
    return _feature_size


def get_rcnn_backbone(models:str = "resnet50") -> torch.nn.Sequential:
    """
    Get the backbone of the R-CNN model.
    :param models: Model name. Default is "resnet50".
    :return: Backbone of the R-CNN model.
    """
    if models == "resnet50":
        _model = tmodels.resnet50(pretrained=True)
    else:
        raise ValueError("Model not supported")

    ### Freeze the layers
    for name, param in _model.named_parameters():
        if name.startswith("layer1") or name.startswith("layer2"):
            param.requires_grad = False

    ### Remove the last two layers and return the backbone
    backbone = torch.nn.Sequential(*list(_model.children())[:-2])
    return backbone


def generate_rp_gt(gt_boxes:list[tuple[tuple[int, int, int, int], int]], image_size:tuple[int, int], num_bg:int = 5) -> list[tuple[tuple[int, int, int, int], int]]:
    """
    Generate the region proposals for the R-CNN model.
    :param gt_boxes: Ground truth boxes. List of tuples of (x1, y1, x2, y2) and class idx.
    :param image_size: Size of the image. Tuple of (width, height).
    :param num_bg: Number of background boxes. Default is 5.
    :return: List of tuples of (x1, y1, x2, y2) and class idx.
    """
    ### Check the parameters are None
    if gt_boxes is None:
        raise ValueError("param \'gt_boxes\' cannot be None")
    if image_size is None:
        raise ValueError("param \'image_size\' cannot be None")
    if num_bg is None:
        raise ValueError("param \'num_bg\' cannot be None")

    ### Generate the positive proposals
    rp_gt = gt_boxes

    ### Generate the background RoI
    added: int = 0
    while added < num_bg:
        ### Generate the random background boxes
        w = np.random.randint(20, image_size[0] // 2)
        h = np.random.randint(20, image_size[1] // 2)
        x1 = np.random.randint(0, image_size[0] - w)
        y1 = np.random.randint(0, image_size[1] - h)
        x2, y2 = x1 + w, y1 + h

        ### Check IoU with the positive boxes
        iou_max: float = 0.0
        for gx1, gy1, gx2, gy2 in [box[0] for box in gt_boxes]:
            ix1 = max(x1, gx1)
            iy1 = max(y1, gy1)
            ix2 = min(x2, gx2)
            iy2 = min(y2, gy2)
            if (ix2 > ix1) and (iy2 > iy1):
                inter_area = (ix2 - ix1) * (iy2 - iy1)
            else:
                inter_area = 0.0
            prop_area = w * h
            gt_area = (gx2 - gx1) * (gy2 - gy1)
            iou = inter_area / float(prop_area + gt_area - inter_area)
            iou_max = max(iou_max, iou)

        ### If IoU is less than .1, add the background box
        if iou_max < 0.1:
            rp_gt.append(((x1, y1, x2, y2), 0))
            added += 1

    return rp_gt