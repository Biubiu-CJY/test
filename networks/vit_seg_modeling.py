# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#改
from .path_util import pjoin

import copy
import logging
import math

#from os.path import join as pjoin



import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, Parameter
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2,DANetHead,CBAM
import torch.nn.functional as F

"""这是最核心的文件，它整合了上述配置和模型定义，
构建了完整的Vision Transformer模型，用于图像分割任务。
文件中定义了多个重要的类，包括用于实现ViT的Attention、Mlp、
Embeddings、Block、Encoder和Transformer类，
以及用于图像分割的解码器DecoderCup和SegmentationHead。
它也包括了模型权重加载的功能，允许从预训练的ViT模型中加载权重以提升性能。"""





logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"
FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

#改

#

class Depthwise_Separable(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depthwise_Separable, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        for rate in rates:
            self.convs.append(nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate))

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Upsample(size=(14, 14), mode='bilinear', align_corners=True)
        )
        self.out_conv = nn.Conv2d(len(rates) * out_channels + out_channels, out_channels, 1)

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        features.append(self.global_pool(x))
        x = torch.cat(features, dim=1)
        x = self.out_conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

        # ASPP module integrated
        self.aspp = ASPP(self.all_head_size, self.all_head_size, [1, 6, 12, 18])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)

        weights = attention_probs if self.vis else None  # Save weights for visualization

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Apply ASPP on the reshaped context layer
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(-1, self.num_attention_heads * self.attention_head_size, 14,
                                           14)  # Reshape assuming 14x14 spatial dimensions
        # context_layer = self.aspp(context_layer)  # Apply ASPP
        context_layer = context_layer.view(*new_context_layer_shape)  # Flatten back to original dimension

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights




# class Attention(nn.Module):#原来的注意力机制
#     def __init__(self, config, vis):
#         super(Attention, self).__init__()
#         self.vis = vis
#         self.num_attention_heads = config.transformer["num_heads"]#head数量
#         self.attention_head_size = int(config.hidden_size / self.num_attention_heads)#head大小
#         self.all_head_size = self.num_attention_heads * self.attention_head_size#计算所有注意力头的总尺寸。
#
#         self.query = Linear(config.hidden_size, self.all_head_size)#dim,dk
#         self.key = Linear(config.hidden_size, self.all_head_size)
#         self.value = Linear(config.hidden_size, self.all_head_size)
#
#         self.out = Linear(config.hidden_size, config.hidden_size)
#         self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
#
#         self.softmax = Softmax(dim=-1)
#
#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         print("self.num_attention_heads",self.num_attention_heads)
#         print("self.attention_head_size",self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)
#
#     def forward(self, hidden_states):
#         print("Attention模块")
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)
#
#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)
#
#         # key_layer = self.attn_dropout(key_layer)  # 这是修改的部分，防止过拟合，dropkey方法
#
#
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#
#         attention_probs = self.softmax(attention_scores) #原代码
#
#         weights = attention_probs if self.vis else None
#         attention_probs = self.attn_dropout(attention_probs)
#
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         attention_output = self.out(context_layer)
#         attention_output = self.proj_dropout(attention_output)
#         # print("weight返回类型encode：", weights.shape)
#         print("attention返回长度：", attention_output.shape)
#         return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])


        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        print(x.shape,"Mlp1：fc1")   #torch.Size([32, 196, 768])
        x = self.fc1(x)
        print(x.shape,"Mlp2：")   #torch.Size([32, 196, 3072])
        x = self.act_fn(x)
        print(x.shape,"Mlp3")   #torch.Size([32, 196, 3072])
        x = self.dropout(x)
        x = self.fc2(x)
        print(x.shape,"Mlp4")   #torch.Size([32, 196, 768])
        x = self.dropout(x)
        print(x.shape,"Mlp5")   #torch.Size([32, 196, 768])
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    position_embeddings: Parameter

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        #改
        # self.DAblock1 = DANetHead(768, 768)
        # self.cbam=CBAM(768)


        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])



    def forward(self, x):

        if self.hybrid:
            x, features = self.hybrid_model(x)
            print("Resnet")
            print(type(features))
        else:
            features = None
        print("Embeding层")
        #print("Embedding处理前：",x.shape)#[16, 1024, 14, 14]
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        #print("patchEmbedding后：", x.shape)#[16, 768, 14, 14]

#改

        # x = self.DAblock1(x)
        # x=self.cbam(x)

        x = x.flatten(2)
        print("flatten后：", x.shape)#flatten后： torch.Size([16, 768, 196])
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        print("transpose后：", x.shape)#transpose后： torch.Size([16, 196, 768])

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        # embeddings = self.dropout(x)
        print("dropout后：", x.shape)
        return embeddings, features


class Block(nn.Module):#Mlp模块
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)#自注意力计算

    def forward(self, x):
        print("Block")
        h = x
        x = self.attention_norm(x)#归一化
        print("输入attention计算前的x：",x.shape)
        x, weights = self.attn(x)#注意力计算
        print("输入attention计算后的x：", x.shape)
        x = x + h

        h = x
        x = self.ffn_norm(x)#LayerNorm
        x = self.ffn(x)#MLP
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        print("EnCoder")
        attn_weights = []
        total_weights=[]
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
                print("vis is true")
        encoded = self.encoder_norm(hidden_states)
        # print("encoded返回类型encode：",type(encoded))
        # print("encoded.shape:",encoded.shape)
        # print("encoded返回类型weights：",type(attn_weights))
        # if len(attn_weights) == 0:
        #     print("列表为空")
        # else:
        #     print("列表非空",len(attn_weights))
        #     ceshi=attn_weights[0]
        #     print(ceshi.shape)
        #     print(attn_weights[0])
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features



class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)



# class DecoderBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             skip_channels=0,
#             use_batchnorm=True,
#     ):
#         super().__init__()
#         self.conv1 = Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.conv2 = Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)

#         #改
#         # self.da = DANetHead(64, 64)
#         # self.da2 = DANetHead(256, 256)
#         # self.da3 = DANetHead(512, 512)
#
#         # self.cbam1=CBAM(64)
#         # self.cbam2 = CBAM(256)
#         # self.cbam3 = CBAM(512)

#     def forward(self, x, skip=None):
#         print(x.shape, "DecoderBlock1")
#         x = self.up(x)
#         print(x.shape, "DecoderBlock2")
#         if skip is not None:#跳跃连接
#             #改
#             # if skip.size(1) and x.size(1) == 64:
#             #     skip = self.da(skip)
#             #
#             # if skip.size(1) and x.size(1) == 256:
#             #     skip = self.da2(skip)
#             #
#             # if skip.size(1) and x.size(1) == 512:
#             #     skip = self.da3(skip)
#
#             # if skip.size(1) and x.size(1) == 64:
#             #     skip = self.cabm1(skip)
#             #
#             # if skip.size(1) and x.size(1) == 256:
#             #     skip = self.cbam2(skip)
#             #
#             # if skip.size(1) and x.size(1) == 512:
#             #     skip = self.cbam3(skip)
# #
#             x = torch.cat([x, skip], dim=1)
#             print(x.shape, "DecoderBlock3_1")
#         print(x.shape, "DecoderBlock3_2")
#         x = self.conv1(x)
#         print(x.shape, "DecoderBlock4")
#         x = self.conv2(x)
#         print(x.shape, "DecoderBlock5")
#         return x

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.DSC1 = Depthwise_Separable(out_channels,out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x, skip=None):
        print(x.shape, "进入解码块")
        x = self.up(x)
        print(x.shape, "上采样后")
        print("skip的类型:",type(skip))
        if skip is not None:#跳跃连接
            x = torch.cat([x, skip], dim=1)
            print(x.shape, "跳跃连接后")
        print(x.shape, "是否跳跃连接后")
        x = self.conv1(x)#通道数变成原来的四分之一，长宽不变
        # x = self.DSC1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        print(x.shape, "conv1后")
        x = self.conv2(x)#张量的尺寸大小不变
        print(x.shape, "conv2后，解码块结束")
        return x

#亚像素卷积
# class DecoderBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             skip_channels=0,
#             use_batchnorm=True,
#     ):
#         super().__init__()
#         # 首先通过卷积层增加通道数为原来的4倍，以适应PixelShuffle
#         self.up_conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1, stride=1)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#
#         self.conv1 = Conv2dReLU(
#             in_channels + skip_channels,  # 这里需要确保channel数目正确
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.conv2 = Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#
#     def forward(self, x, skip=None):
#         x = self.up_conv(x)  # 先进行卷积扩展通道
#         x = self.pixel_shuffle(x)  # 进行PixelShuffle上采样
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x






class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=True):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(64)

        # Define ReLU activation layers for each feature
        # Note: ReLU layers do not have parameters, so you could technically use one layer for all.
        self.relu = nn.ReLU(inplace= True)

    def forward(self, x):
        print(x.shape, "开始")
        bc = x.shape[0]
        if x.size()[1] == 1:#如果是单通道就，重复三个单通道，变成三通道
            x = x.repeat(1,3,1,1)
            print(x.shape,"重复3次单通道")
        print(x.shape,"VisionTransformer1_2")
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)#调用Transformer模块，计算qkv

        print("features0.shape", features[0].shape)
        print("features1.shape", features[1].shape)
        print("features2.shape", features[2].shape)
        # print(attn_weights[0].shape)
        # print(len(attn_weights))
        # concatenated_weights = torch.cat(attn_weights, dim=1)
        #
        #
        # # 输出拼接后的张量形状，以确保形状是 [16, 12*len(attn_weights), 196, 196]
        # print("Concatenated Weights Shape:", concatenated_weights.shape)
        #
        # collapsed_weights1 = concatenated_weights.sum(dim=3)
        #
        # print("collapsed_weights1.shape",collapsed_weights1.shape)
        # # 使用 reshape 进行变形
        # reshaped_weights = collapsed_weights1.reshape(bc, 144, 14, 14)
        #
        # def resize_weights(weights, size):
        #     # 使用双线性插值调整weights的尺寸
        #     return F.interpolate(weights, size=size, mode='bilinear', align_corners=False)
        #
        # # 假设weights的初始尺寸为[16, 144, 14, 14]
        # weights_resized_28 = resize_weights(reshaped_weights, (28, 28))
        # weights_resized_56 = resize_weights(reshaped_weights, (56, 56))
        # weights_resized_112 = resize_weights(reshaped_weights, (112, 112))
        # print("weights_resized_28",weights_resized_28.shape)
        # print("weights_resized_56", weights_resized_56.shape)
        # print("weights_resized_112", weights_resized_112.shape)
        # # 定义 Conv2d 层以扩展通道维度
        # conv2d1 = nn.Conv2d(in_channels=144, out_channels=512, kernel_size=1)
        # conv2d2 = nn.Conv2d(in_channels=144, out_channels=256, kernel_size=1)
        # conv2d3 = nn.Conv2d(in_channels=144, out_channels=64, kernel_size=1)
        #
        # # 将卷积层移至 GPU
        # conv2d1 = conv2d1.cuda()
        # conv2d2 = conv2d2.cuda()
        # conv2d3 = conv2d3.cuda()
        #
        # # 应用卷积调整通道维度
        # weights28 = conv2d1(weights_resized_28.cuda())
        # weights56 = conv2d2(weights_resized_56.cuda())
        # weights112 = conv2d3(weights_resized_112.cuda())
        # features[0]=features[0]+weights28
        # features[1]=features[1]+weights56
        # features[2]=features[2]+weights112



        # # 将列表中的张量堆叠成一个新的张量
        # stacked_attn_weights = torch.stack(attn_weights)
        # # 对堆叠后的张量在新的维度（第0维，即列表的长度维度）进行求和
        # summed_attn_weights = torch.sum(stacked_attn_weights, dim=0)
        # print("尺寸",summed_attn_weights.shape)
        # # 对张量进行列累加，即沿着第三个维度（dim=2）进行求和
        # collapsed_weights = summed_attn_weights.sum(dim=3)
        # # 输出处理后的张量形状，检查是否为 ([16, 12, 196])
        # print("Collapsed Weights Shape:", collapsed_weights.shape)
        # # 创建一个 1D 卷积层，输入通道为 12，输出通道为 768，卷积核大小为 1
        # conv1d = nn.Conv1d(in_channels=12, out_channels=768, kernel_size=1)
        # # 将卷积层移动到相同的设备上（GPU）
        # conv1d = conv1d.cuda()
        #
        # # 应用卷积层到 collapsed_weights，不改变最后一维
        # expanded_weights = conv1d(collapsed_weights)
        #
        # print(expanded_weights.shape)  # 输出的形状应为 [16, 768, 196]
        # # 使用 reshape 进行变形
        # reshaped_weights = expanded_weights.reshape(16, 768, 14, 14)
        # print(reshaped_weights.shape)  # 输出的形状应为 [16, 768, 14, 14]

        # def resize_feature(weights, size):
        #     # 使用双线性插值调整weights的尺寸
        #     return F.interpolate(weights, size=size, mode='bilinear', align_corners=False)
        # features1 = resize_feature(features[0], (56, 56))
        # features1= resize_feature(features1, (112, 112))
        # features2 = resize_feature(features[1], (112, 112))
        # features3=features[2]
        #
        #
        # print("features1.shape", features1.shape)
        # print("features2.shape", features2.shape)
        # print("features3.shape", features3.shape)
        #
        #
        # conv2d1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
        # conv2d2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        # conv2d3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        # conv2d6 = nn.Conv2d(in_channels=384,out_channels=64, kernel_size=1)
        # conv2d5 = nn.Conv2d(in_channels=384,out_channels=256, kernel_size=1)
        # conv2d4 = nn.Conv2d(in_channels=384,out_channels=512, kernel_size=1)
        #
        # # 将卷积层移至 GPU
        # conv2d1 = conv2d1.cuda()
        # conv2d2 = conv2d2.cuda()
        # conv2d3 = conv2d3.cuda()
        # conv2d4 = conv2d4.cuda()
        # conv2d5 = conv2d5.cuda()
        # conv2d6 = conv2d6.cuda()
        #
        # # 应用卷积调整通道维度
        # # 应用卷积调整通道维度
        # features1= conv2d1(features1.cuda())
        # features2= conv2d2(features2.cuda())
        # features3 = conv2d3(features3.cuda())
        #
        #
        #
        # fused_features = torch.cat([features1, features2, features3], dim=1)
        #
        #
        # features_3=fused_features#([1, 384, 112, 112])
        # features_2=resize_feature(fused_features, (56, 56))#([1, 384, 56,56])
        # features_1=resize_feature(fused_features, (28,28))#([1, 384, 28,28])
        #
        # features_1 = conv2d4(features_1.cuda())
        # features_1 = self.relu(self.bn1(features_1))
        # features_2=conv2d5(features_2.cuda())
        # features_2 = self.relu(self.bn2(features_2))
        # features_3 = conv2d6(features_3.cuda())
        # features_3 = self.relu(self.bn3(features_3))
        #
        #
        # print("features_1.shape", features_1.shape)
        # print("features_2.shape", features_2.shape)
        # print("features_3.shape", features_3.shape)
        #
        # features[0]=features_1
        # features[1]=features_2
        # features[2]=features_3



        # features4 = conv2d4(fused_features.cuda())
        # features5 = conv2d5(fused_features.cuda())
        # features6 = conv2d6(fused_features.cuda())

        # print("fused_features .shape", fused_features.shape)
        #
        # print("features1.shape", features1.shape)
        # print("features2.shape", features2.shape)
        # print("features3.shape", features3.shape)
        # print("features4.shape", features4.shape)
        # print("features5.shape", features5.shape)
        # print("features6.shape", features6.shape)


        # print(" features2_upsampled", features[1].shape)
        # print(x.shape,"Transformer类之后")
        x = self.decoder(x, features)
        print(x.shape,"Decoder后")
        logits = self.segmentation_head(x)
        print(logits.shape,"分割头后")
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


