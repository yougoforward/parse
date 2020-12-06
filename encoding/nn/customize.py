##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, BCELoss, CrossEntropyLoss

from torch.autograd import Variable
from itertools import filterfalse as ifilterfalse

torch_ver = torch.__version__[:3]

__all__ = ['SegmentationLosses', 'SegmentationLosses_parse', 'PyramidPooling', 'JPU', 'JPU_X', 'Mean', 'ASPPModule', 'SEModule']

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 size_average=True, ignore_index=-1, reduction='mean'):
        super(SegmentationLosses, self).__init__(weight, ignore_index=ignore_index, reduction=reduction)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, reduction=reduction)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

class SegmentationLosses_parse(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 size_average=True, ignore_index=-1, reduction='mean'):
        super(SegmentationLosses_parse, self).__init__(weight, ignore_index=ignore_index, reduction=reduction)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, reduction=reduction)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.only_present = True
    def forward(self, *inputs):
        # print(len(list(inputs)))
        part, half, full, aux, targets = tuple(inputs)
        preds = [part, half, full, aux]
        print(targets.size())
        targets = torch.split(targets, 1, dim=3)
        h, w = targets[0].size(1), targets[0].size(2)
        #part seg loss final
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        #ce loss
        loss_ce = self.criterion(pred, targets[0])
        pred = F.softmax(input=pred, dim=1)
        #lovasz loss
        lovasz_loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        loss_final = (lovasz_loss + loss_ce)
        # loss_final = lovasz_loss
        
        #half seg loss final
        pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        #ce loss
        loss_ce = self.criterion(pred, targets[1].long())
        pred = F.softmax(input=pred, dim=1)
        #lovasz loss
        lovasz_loss = lovasz_softmax_flat(*flatten_probas(pred, targets[1], self.ignore_index), only_present=self.only_present)
        loss_final_hb = (lovasz_loss + loss_ce)
        # loss_final_hb = lovasz_loss

        #full seg loss final
        pred = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        #ce loss
        loss_ce = self.criterion(pred, targets[2].long())
        pred = F.softmax(input=pred, dim=1)
        #lovasz loss
        lovasz_loss = lovasz_softmax_flat(*flatten_probas(pred, targets[2], self.ignore_index), only_present=self.only_present)
        loss_final_fb = (lovasz_loss + loss_ce)
        # loss_final_fb = lovasz_loss
        
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])

        return (loss_final+0.2*loss_final_hb+0.2*loss_final_fb) + 0.4 * loss_dsn

    
class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class ASPPModule(nn.Module):
    """ASPP"""

    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(ASPPModule, self).__init__()

        self.dilation_0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True), 
                                SEModule(out_channels, reduction=16))

        self.dilation_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True),
                                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True),
                                SEModule(out_channels, reduction=16))

        self.dilation_2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True),
                                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True),
                                SEModule(out_channels, reduction=16))

        self.dilation_3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True),
                                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True),
                                SEModule(out_channels, reduction=16))                           
        self.psaa_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                    norm_layer(out_channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(out_channels, 4, 1, bias=True),
                                    nn.Sigmoid())  

        self.project = nn.Sequential(nn.Conv2d(in_channels=4*out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(out_channels),
                      nn.ReLU(True))

    def forward(self, x):
        # parallel branch
        feat0 = self.dilation_0(x)
        feat1 = self.dilation_1(x)
        feat2 = self.dilation_2(x)
        feat3 = self.dilation_3(x)
        # psaa
        psaa_att = self.psaa_conv(x)
        psaa_att_list = torch.split(psaa_att, 1, dim=1)
        y2 = torch.cat((psaa_att_list[0] * feat0, psaa_att_list[1] * feat1, psaa_att_list[2] * feat2, psaa_att_list[3] * feat3), 1)
        out = self.project(y2)
        return out

class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return inputs[0], inputs[1], inputs[2], feat


class JUM(nn.Module):
    def __init__(self, in_channels, width, dilation, norm_layer, up_kwargs):
        super(JUM, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv_l = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        norm_layer = lambda n_channels: nn.GroupNorm(32, n_channels)
        self.dilation1 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=dilation, dilation=dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=2*dilation, dilation=2*dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=4*dilation, dilation=4*dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, x_l, x_h):
        feats = [self.conv_l(x_l), self.conv_h(x_h)]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([feats[-2], self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)

        return feat

class JPU_X(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU_X, self).__init__()
        self.jum_1 = JUM(in_channels[:2], width//2, 1, norm_layer, up_kwargs)
        self.jum_2 = JUM(in_channels[1:], width, 2, norm_layer, up_kwargs)

    def forward(self, *inputs):
        feat = self.jum_1(inputs[2], inputs[1])
        feat = self.jum_2(inputs[3], feat)

        return inputs[0], inputs[1], inputs[2], feat


class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)
    
    
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat_ori(*flatten_probas_ori(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat_ori(*flatten_probas_ori(probas, labels, ignore), classes=classes)
    return loss

def lovasz_softmax_flat_ori(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas_ori(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def lovasz_softmax_flat(preds, targets, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      :param preds: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      :param targets: [P] Tensor, ground truth labels (between 0 and C - 1)
      :param only_present: average only on classes present in ground truth
    """
    if preds.numel() == 0:
        # only void pixels, the gradients should be 0
        return preds * 0.

    C = preds.size(1)
    losses = []
    for c in range(C):
        fg = (targets == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - preds[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probas(preds, targets, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = preds.size()
    preds = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    targets = targets.contiguous().view(-1)
    if ignore is None:
        return preds, targets
    valid = (targets != ignore)
    vprobas = preds[valid.nonzero().squeeze()]
    vlabels = targets[valid]
    return vprobas, vlabels

# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def mean(l, ignore_nan=True, empty=0):
    """
    nan mean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def isnan(x):
    return x != x
