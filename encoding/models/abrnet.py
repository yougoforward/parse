###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division

import torch
import torch.nn as nn

from torch.nn.functional import interpolate

from .base import BaseNet
from .fcn import FCNHead
from ..nn import ASPPModule, SEModule

__all__ = ['abrnet', 'get_abrnet']

class abrnet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(abrnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = abrnetHead(2048, nclass, norm_layer)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)

        x = self.head(c2, c2, c3, c4)

        outputs = [interpolate(xi, imsize, **self._up_kwargs) for xi in x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)

        
class abrnetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(abrnetHead, self).__init__()
        inter_channels = in_channels // 4
        self.layer5 = ASPPModule(2048, 512, norm_layer, up_kwargs)
        self.layerp = DecoderModule(norm_layer, num_classes)
        self.layerh = AlphaDecoder(norm_layer, hbody_cls)
        self.layerf = AlphaDecoder(norm_layer, fbody_cls)

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, c1, c2, c3, c4):
        context = self.layer5(c4)

        x_seg, xt_fea = self.layerp(context, c2, c1)
        alpha_hb = self.layerh(context, c2)
        alpha_fb = self.layerf(context, c1)
        return [x_seg, alpha_hb, alpha_fb]


def get_abrnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = abrnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('abrnet_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model


class DecoderModule(nn.Module):

    def __init__(self, norm_layer, num_classes):
        super(DecoderModule, self).__init__()
        self.conv_m = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                   norm_layer(512), nn.ReLU(True))
        self.conv1 = nn.Sequential(
                                   nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                                   norm_layer(256), nn.ReLU(True)
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                   norm_layer(256), nn.ReLU(True))

        self.conv3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                   norm_layer(256), nn.ReLU(True),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
                                   norm_layer(256), nn.ReLU(True),
                                   SEModule(256, reduction=16))

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xm, xl):
        _, _, h, w = xm.size()

        xt_fea = self.conv1(F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True) + self.conv_m(xm))
        _, _, th, tw = xl.size()
        xt = F.interpolate(xt_fea, size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x_fea = self.conv3(xt+xl)
        x_seg = self.conv4(x_fea)
        return x_seg, xt_fea


class AlphaDecoder(nn.Module):
    def __init__(self, norm_layer, cls):
        super(AlphaDecoder, self).__init__()
        self.conv_m = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                   norm_layer(512), nn.ReLU(True))
        self.conv1 = nn.Sequential(
                                   nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                                   norm_layer(256), nn.ReLU(True),
                                   SEModule(256, reduction=16) 
                                   )
                                   
        self.cls_hb = nn.Conv2d(256, cls, kernel_size=1, padding=0, stride=1, bias=True)
        self.alpha_hb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()
        xfuse = self.conv1(F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + self.conv_m(skip))
        output = self.conv1(xfuse)
        output = self.cls_hb(output)
        return output
