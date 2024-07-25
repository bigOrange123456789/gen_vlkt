# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):#input:[ResNet(...) True 2048 False]
        # nn.Module: <class 'torch.nn.modules.module.Module'>
        super().__init__()
        for name, parameter in backbone.named_parameters():# 遍历主干网络每一层 的 名称和参数
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name: # 为啥？只训练layer2、layer3、layer4
                parameter.requires_grad_(False) #在前向传播完成后，这一参数是否在显存中保留梯度，等待优化器执行optim.step()更新参数。
        if return_interm_layers: # return_interm_layers : False
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"} # 不返回中间层
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers) # IntermediateLayerGetter这个封装函数的作用是获取需要层的输出结果
        self.num_channels = num_channels # num_channels: 2048

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, # name:resnet50,
                 train_backbone: bool, # 训练主干=True,
                 return_interm_layers: bool, # 返回中间层=False,
                 dilation: bool):#  膨胀=False
        backbone = getattr(torchvision.models, name)(#[torchvision.models=<module 'torchvision.models'>  , name=resnet50]
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, 
            norm_layer=FrozenBatchNorm2d) # FrozenBatchNorm2d: <class 'models.backbone.FrozenBatchNorm2d'>
        # 上面的操作是获取torchvision.models中名称为name的网络结构 : <generator object Module.named_parameters at 0x0000017937BE4840>
        # backbone : ResNet(...)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048 # num_channels: 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers) # super().__init__() 就是调用父类的init方法


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding) # 在最后添加一个余弦嵌入层

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args) # position_embedding: PositionEmbeddingSine()
    train_backbone = args.lr_backbone > 0 # [args.lr_backbone: 1e-05 ,train_backbone: True]
    return_interm_layers = args.masks # args.masks: False
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation) #[backbone: Backbone(..) , input:(resnet50 True False False)] 
    model = Joiner(backbone, position_embedding) # [model: Joiner(...) , position_embedding: PositionEmbeddingSine()]
    model.num_channels = backbone.num_channels # num_channels: 2048
    return model
