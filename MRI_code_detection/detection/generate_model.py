import torchvision
import torch
import torch.nn.functional as F
import config
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops
from collections import OrderedDict
from .layer_getter import IntermediateLayerGetter
from .faster_rcnn import FasterRCNN
from . import resnet

import cv2
import numpy as np

def heat_map(list_features):
    
    # list_features = list_features.cpu().detach().numpy() 
    print(list_features.keys())
    for heatmaps in list_features['pool']:
        print(type(heatmaps))
        heatmaps = heatmaps.cpu().detach().numpy()
        for heatmap in heatmaps:
            v_min = heatmap.min()
            v_max = heatmap.max()
            heatmap = (heatmap - v_min) / max((v_max - v_min),1e-10)
            heatmap = cv2.resize(heatmap,(512,512)) * 255
    
            heatmap = heatmap.astype(np.uint8)
            # cv2.imshow("heatmap1",heatmap)
            cv2.imwrite("/home/myy/jingzhui/MRI/MRI_data_prepare/show_pic/heatmap1.png", heatmap)
            # cv2.waitKey()
            heatmap2 = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # heatmap2 = cv2.cvtColor(heatmap,cv2.COLOR_GRAY2BGR)
            # cv2.imshow("heatmap2", heatmap2)
            cv2.imwrite("/home/myy/jingzhui/MRI/MRI_data_prepare/show_pic/heatmap2.png", heatmap2)
            # cv2.waitKey()
            # superimposed_img = heatmap2 * 0.4 + img * 0.6
            # superimposed_img = np.clip(superimposed_img,0,255).astype(np.uint8)
            # # cv2.imshow("superimposed_img", superimposed_img)
            # cv2.imwrite("/home/myy/jingzhui/MRI/MRI_data_prepare/show_pic/superimposed_img.png", superimposed_img)
            # cv2.waitKey()

class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        # heat_map(x)
        return x


def load_weight(backbone):
    backbone_dict = backbone.state_dict()
    pretrain = torchvision.models.resnet50(pretrained=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    pretrain_dict = pretrain.state_dict()

    update_dict = {k: v for k, v in pretrain_dict.items() if k in backbone_dict}
    backbone_dict.update(update_dict)

    for x in pretrain_dict.keys():
        if 'conv' in x:
            y = x.replace('.weight', '.feature_conv.weight')
            if y in backbone_dict:
                backbone_dict[y] = pretrain_dict[x]
            if '.bias' in x:
                y = x.replace('.bias', '.feature_conv.bias')
                if y in backbone_dict:
                    backbone_dict[y] = pretrain_dict[x]

    backbone.load_state_dict(backbone_dict)
    return backbone


def resnet_fpn_backbone(pretrained=False, deformable=False):
    backbone = resnet.resnet50(
        pretrained=pretrained,
        deformable=deformable,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = load_weight(backbone)

    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


def faster_rcnn(num_classes=1000, deformable=False):
    backbone = resnet_fpn_backbone(deformable=deformable)

    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 0.8, 1.3),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                    output_size=7,
                                    sampling_ratio=2)

    image_mean = config.image_mean
    image_std = config.image_std
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       image_mean=image_mean,
                       image_std=image_std)
    return model
