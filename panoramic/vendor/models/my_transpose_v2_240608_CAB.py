import logging
import math

import copy
from functools import partial
from collections import OrderedDict

import cv2
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Optional, List

from .vision_transformer import VisionTransformer

# from .DeiT import deit_base_patch16_384 as deit_base_patch16_384

logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.1


class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleDecoderBlock, self).__init__()

        # 定义一个上采样层
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        # 定义一个 3x3 卷积层
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 定义一个 ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 上采样
        x = self.upsample(x)

        # ReLU 激活
        x = self.relu(x)

        # 3x3 卷积
        x = self.conv3x3(x)

        return x


class ClassicDecoderBlock(nn.Module):
    def __init__(self, ch1, ch2, ch3, ch_out, layers):
        super(ClassicDecoderBlock, self).__init__()

        self.layers = layers

        # 定义第一个反卷积块
        self.deconv1 = nn.ConvTranspose2d(ch1, ch2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(ch2)
        self.relu1 = nn.ReLU(inplace=True)

        # 定义第二个反卷积块
        self.deconv2 = nn.ConvTranspose2d(ch2, ch3, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ch3)
        self.relu2 = nn.ReLU(inplace=True)

        # 定义第三个反卷积块
        self.deconv3 = nn.ConvTranspose2d(ch3, ch_out, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ch_out)
        self.relu3 = nn.ReLU(inplace=True)

        # # 定义 1x1 卷积层用于生成热图
        # self.conv1x1 = nn.Conv2d(ch_out, ch_out, kernel_size=1)

    def forward(self, x):

        # 第一个反卷积块
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 第二个反卷积块
        if self.layers == 2 or self.layers == 3:
            x = self.deconv2(x)
            x = self.bn2(x)
            x = self.relu2(x)

            # 第三个反卷积块
            if self.layers == 3:
                x = self.deconv3(x)
                x = self.bn3(x)
                x = self.relu3(x)

        # # 1x1 卷积层
        # x = self.conv1x1(x)

        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


def vit_base_patch16_384_in21k(out_chal: int = 256, out_size=12, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(image_size=384,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=out_chal * out_size * out_size)

    return model


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in

        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class MBConvConfig:
    """EfficientNet Building block configuration"""

    def __init__(self, expand_ratio: float, kernel: int, stride: int, in_ch: int, out_ch: int, layers: int,
                 use_se: bool, fused: bool, act=nn.SiLU, norm_layer=nn.BatchNorm2d):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers
        self.act = act
        self.norm_layer = norm_layer
        self.use_se = use_se
        self.fused = fused

    @staticmethod
    def adjust_channels(channel, factor, divisible=8):
        new_channel = channel * factor
        divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel


class MBConv(nn.Module):
    """EfficientNet main building blocks

    :arg
        - c: MBConvConfig instance
        - sd_prob: stochastic path probability
    """

    def __init__(self, c, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
        block = []

        if c.expand_ratio == 1:
            block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
        elif c.fused:
            block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
            block.append(('fused_point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))
        else:
            block.append(('linear_bottleneck', ConvBNAct(c.in_ch, inter_channel, 1, 1, 1, c.norm_layer, c.act)))
            block.append(('depth_wise',
                          ConvBNAct(inter_channel, inter_channel, c.kernel, c.stride, inter_channel, c.norm_layer,
                                    c.act)))
            block.append(('se', SEUnit(inter_channel, 4 * c.expand_ratio)))
            block.append(('point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")
        self.ghost = GhostModule(c.in_ch, c.out_ch)

    def forward(self, x):
        g_out = self.ghost(x)
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out) + g_out
        return out


class ConvBNAct(nn.Sequential):
    """Convolution-Normalization-Activation Module"""

    def __init__(self, in_channel, out_channel, kernel_size, stride, groups, norm_layer, act, conv_layer=nn.Conv2d):
        super(ConvBNAct, self).__init__(
            conv_layer(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                       groups=groups, bias=False),
            norm_layer(out_channel),
            act()
        )


class SEUnit(nn.Module):
    """Squeeze-Excitation Unit

    paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper

    """

    def __init__(self, in_channel, reduction_ratio=4, act1=partial(nn.SiLU, inplace=True), act2=nn.Sigmoid):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))


class StochasticDepth(nn.Module):
    """StochasticDepth

    paper: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39

    :arg
        - prob: Probability of dying
        - mode: "row" or "all". "row" means that each row survives with different probability
    """

    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * torch.empty(shape).bernoulli_(self.survival).div_(self.survival).to(x.device)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask, pos=pos,
                               src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # 殘差
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # 殘差
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransPoseV2(nn.Module):
    def __init__(self, layer_infos, out_channels=1280, nclass=6, dropout=0.2, stochastic_depth=0.0,
                 block=MBConv, norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU):
        super(TransPoseV2, self).__init__()
        # self.inplanes = 64
        self.inplanes = 128
        self.deconv_with_bias = False

        # get_sd_prob 設定
        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        # self.stem 設定
        self.in_channel = layer_infos[0].in_ch  # in_ch代表輸入config中in的那一直行
        # self.final_stage_channel = layer_infos[-1].out_ch

        # self.head 設定
        # self.head_out_channels = out_channels
        self.norm_layer = norm_layer
        self.act = act_layer

        # ConvBNAct(in_channel, out_channel, kernel_size, stride, groups, norm_layer, act, conv_layer=nn.Conv2d)
        self.stem = ConvBNAct(3, self.in_channel, 3, 2, 1, self.norm_layer, self.act)

        self.blocks0 = nn.Sequential(*self.make_stages(layer_infos, block, stage_num=0))
        self.blocks1 = nn.Sequential(*self.make_stages(layer_infos, block, stage_num=1))
        self.blocks2 = nn.Sequential(*self.make_stages(layer_infos, block, stage_num=2))
        self.blocks3 = nn.Sequential(*self.make_stages(layer_infos, block, stage_num=3))
        self.blocks4 = nn.Sequential(*self.make_stages(layer_infos, block, stage_num=4))
        self.blocks5 = nn.Sequential(*self.make_stages(layer_infos, block, stage_num=5))
        self.blocks6 = nn.Sequential(*self.make_stages(layer_infos, block, stage_num=6))

        self.head4 = nn.Sequential(OrderedDict([
            ('bottleneck', ConvBNAct((layer_infos[-6].out_ch), out_channels, 1, 1, 1, self.norm_layer, self.act)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('dropout', nn.Dropout(p=dropout, inplace=True)),
            ('classifier', nn.Linear(out_channels, nclass * 2) if nclass * 2 else nn.Identity())
        ]))

        self.head3 = nn.Sequential(OrderedDict([
            ('bottleneck', ConvBNAct((layer_infos[-4].out_ch), out_channels, 1, 1, 1, self.norm_layer, self.act)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('dropout', nn.Dropout(p=dropout, inplace=True)),
            ('classifier', nn.Linear(out_channels, nclass * 2) if nclass * 2 else nn.Identity())
        ]))

        self.head2 = nn.Sequential(OrderedDict([
            ('bottleneck', ConvBNAct((layer_infos[-2].out_ch), out_channels, 1, 1, 1, self.norm_layer, self.act)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('dropout', nn.Dropout(p=dropout, inplace=True)),
            ('classifier', nn.Linear(out_channels, nclass * 2) if nclass * 2 else nn.Identity())
        ]))

        self.head = nn.Sequential(OrderedDict([
            ('bottleneck', ConvBNAct(layer_infos[-1].out_ch, out_channels, 1, 1, 1, self.norm_layer, self.act)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('dropout', nn.Dropout(p=dropout, inplace=True)),
            ('classifier', nn.Linear(out_channels, nclass * 2) if nclass * 2 else nn.Identity())
        ]))

        ###########################################################################################################

        self.sigmoid = nn.Sigmoid()

        self.transformer = vit_base_patch16_384_in21k(out_chal=160, out_size=24)
        drop_rate = 0.2
        self.drop = nn.Dropout2d(drop_rate)
        # self.up1 = Up(160,80)
        # self.up2 = Up(80,48)

        # self.up1 = ClassicDecoderBlock(160,80,48)
        # self.up1 = SimpleDecoderBlock(160, 6)

        # self.up6 = ClassicDecoderBlock(256,160,80,48,3)
        # self.up5 = ClassicDecoderBlock(160,80,48,48,2)
        # self.up3 = ClassicDecoderBlock(80,48,48,48,1)

        self.up6 = ClassicDecoderBlock(256, 160, 80, 80, 2)
        self.up5 = ClassicDecoderBlock(160, 80, 80, 80, 1)
        self.up3 = ClassicDecoderBlock(240, 48, 48, 48, 1)

        d_model = 64
        dim_feedforward = 1024
        encoder_layers_num = 3  # transformer重複幾層
        n_head = 8
        pos_embedding_type = 'sine'
        w, h = [384, 384]

        # self.reduce = nn.Conv2d(self.inplanes, d_model, kernel_size=1, bias=False)
        self.pos_embedding3 = self._make_position_embedding(pe_w=48, pe_h=48, d_model=240, pe_type=pos_embedding_type)
        self.pos_embedding5 = self._make_position_embedding(pe_w=24, pe_h=24, d_model=160, pe_type=pos_embedding_type)
        self.pos_embedding6 = self._make_position_embedding(pe_w=12, pe_h=12, d_model=256, pe_type=pos_embedding_type)
        # self.final_pos_embedding = self._make_position_embedding(w, h, d_model=6, pe_type=pos_embedding_type)

        self.global_encoder3 = self._make_global_encoder(240, n_head, dim_feedforward, encoder_layers_num)
        self.global_encoder5 = self._make_global_encoder(160, n_head, dim_feedforward, encoder_layers_num)
        self.global_encoder6 = self._make_global_encoder(256, n_head, dim_feedforward, encoder_layers_num)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            activation='relu',
            return_atten_map=False
        )
        self.global_encoder = TransformerEncoder(
            encoder_layer,
            encoder_layers_num,
            return_atten_map=False
        )

        # final_encoder_layer = TransformerEncoderLayer(
        #     6,
        #     nhead=2,
        #     dim_feedforward=dim_feedforward,
        #     activation='relu',
        #     return_atten_map=False
        # )
        # self.final_encoder = TransformerEncoder(
        #     final_encoder_layer,
        #     num_layers = 1,
        #     return_atten_map=False
        # )

        # self.deconv_layers = self._make_deconv_layer(
        #     extra.NUM_DECONV_LAYERS,   # 1
        #     extra.NUM_DECONV_FILTERS,  # [d_model]
        #     extra.NUM_DECONV_KERNELS,  # [4]
        # )

        # self.deconv_layers = self._make_deconv_layer(1,[d_model],[4])
        # self.deconv_layers3 = self._make_deconv_layer(1,[128],[4])
        # self.deconv_layers4 = self._make_deconv_layer(1,[160],[4])
        # self.deconv_layers5 = self._make_deconv_layer(1,[64],[4])

        # self.final_layer = nn.Conv2d(
        #     in_channels=d_model,
        #     out_channels=nclass,     # 關鍵點數量
        #     kernel_size=1,
        #     stride=1,
        # )

        self.final_layer = nn.Conv2d(
            in_channels=48,
            out_channels=nclass,  # 關鍵點數量
            kernel_size=1,
            stride=1,
        )

        # self.final_layer3 = nn.Conv2d(
        #     in_channels=80,
        #     out_channels=nclass,     # 關鍵點數量
        #     kernel_size=1,
        #     stride=1,
        # )

        # self.final_layer5 = nn.Conv2d(
        #     in_channels=160,
        #     out_channels=nclass,  # 關鍵點數量
        #     kernel_size=1,
        #     stride=1,
        # )

        # self.final_layer6 = nn.Conv2d(
        #     in_channels=256,
        #     out_channels=nclass,  # 關鍵點數量
        #     kernel_size=1,
        #     stride=1,
        # )

        # self.fusion = BiFusion_block(ch_1=6, ch_2=6, r_2=2, ch_int=6, ch_out=6, drop_rate=drop_rate/2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_c = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=160, ch_2=160, r_2=2, ch_int=160, ch_out=160, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=256, out_ch=160, in_ch2=160, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=80, ch_2=80, r_2=1, ch_int=80, ch_out=80, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(160, 80, 80, attn=True)

        # self.drop = nn.Dropout2d(drop_rate)

    def _make_global_encoder(self, d_model, n_head, dim_feedforward, encoder_layers_num):
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            activation='relu',
            return_atten_map=False
        )

        global_encoder = TransformerEncoder(
            encoder_layer,
            encoder_layers_num,
            return_atten_map=False
        )
        return global_encoder

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_position_embedding(self, pe_w, pe_h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            logger.info("==> Without any PositionEmbedding~")
            return None
        else:
            with torch.no_grad():
                # pe_h = h // 8
                # pe_w = w // 8
                length = pe_h * pe_w
            if pe_type == 'learnable':
                logger.info("==> Add Learnable PositionEmbedding~")
                return nn.Parameter(
                    torch.randn(length, 1, d_model))
            else:
                # logger.info("==> Add Sine PositionEmbedding~")
                return nn.Parameter(
                    self._make_sine_position_embedding(d_model, pe_w, pe_h),
                    requires_grad=False)

    def _make_sine_position_embedding(self, d_model, pe_w, pe_h, temperature=10000,
                                      scale=2 * math.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 384 // 8 , 192 // 8 #self.pe_h, self.pe_w
        w, h = pe_w, pe_h
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def make_stages(self, layer_infos, block, stage_num):
        layers = []
        i = 0
        for layer_info in layer_infos:
            if i == stage_num:
                layer = self.make_layers(copy.copy(layer_info), block)
                layers.extend(layer)
                return layers
            else:
                i += 1
        return layers

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            # print(i)
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))  # block = MBConv
            # print("layer_info.out_ch", layer_info.out_ch)
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def forward(self, x):
        # print("x.shape", x.shape)

        x = self.stem(x)
        x = self.blocks0(x)
        x1 = self.blocks1(x)
        x2 = self.blocks2(x1)
        x3 = self.blocks3(x2)
        x4 = self.blocks4(x3)
        x5 = self.blocks5(x4)
        x6 = self.blocks6(x5)

        x_head = self.head(x6)

        x5_up = self.up5(x5)
        x6_up = self.up6(x6)

        x3_1 = torch.cat((x3, x5_up, x6_up), dim=1)

        # transpose3分支
        bs, c, h, w = x3_1.shape
        x3_1 = x3_1.flatten(2).permute(2, 0, 1)
        x3_1 = self.global_encoder3(x3_1, pos=self.pos_embedding3)
        x3_1 = x3_1.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        x3_1 = self.up3(x3_1)

        cat_x = self.final_layer(x3_1)
        cat_x = self.sigmoid(cat_x)

        return x_head, cat_x

    def change_dropout_rate(self, p):
        self.head[-2] = nn.Dropout(p=p, inplace=True)


def efficientnet_v2_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)


# 只留下(12,12) (24,24)兩層 使用v2 transpose雙輸出
def get_yoshen_transpose_v2_240608(nclass=6, dropout=0.1, stochastic_depth=0.2, **kwargs):
    layer_configs = [
        # e k  s  in  out xN  se   fused
        (1, 3, 1, 24, 24, 2, False, True),
        (4, 3, 2, 24, 48, 4, False, True),
        (4, 3, 2, 48, 64, 4, False, True),
        (4, 3, 1, 64, 80, 4, True, False),
        (4, 3, 2, 80, 128, 6, True, False),
        (6, 3, 1, 128, 160, 9, True, False),
        (6, 3, 2, 160, 256, 15, True, False),
    ]
    residual_config = []
    for layer_config in layer_configs:
        # print("layer_config", layer_config)
        residual_config.append(MBConvConfig(*layer_config))

        # MBConvConfig 會定義 expand_ratio: float, kernel: int, stride: int, in_ch: int, out_ch: int, layers: int, use_se: bool, fused: bool, act=nn.SiLU, norm_layer=nn.BatchNorm2d):
        # 對應到 layer_config (1, 3, 1, 24, 24, 2, False, True)

    # print("residual_config", residual_config[1])
    model = TransPoseV2(residual_config, out_channels=1280, nclass=nclass, dropout=dropout,
                        stochastic_depth=stochastic_depth, block=MBConv, norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU)
    efficientnet_v2_init(model)

    # if pretrained:
    #     load_from_zoo(model, model_name)

    return model