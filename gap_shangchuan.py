# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmpretrain.registry import MODELS

@MODELS.register_module()
class GlobalAveragePooling_shangchuan(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(GlobalAveragePooling_shangchuan, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'

        # ========自定义参数========
        # 定义 DCT 高度和宽度
        dct_h, dct_w = 7, 7  # 通常是 7x7 的 DCT 块
        frequency_branches = 16  # 频率分支数，必须是 [1, 2, 4, 8, 16, 32] 之一
        frequency_selection = 'top'  # 选择 'top' 频率，应与 get_freq_indices 函数支持的选项匹配
        reduction = 16  # reduction 参数用于控制通道数的缩减，通常设置为 16
        # ========自定义参数========
        # 初始化 MultiFrequencyChannelAttention 模块
        self.mfca = MultiFrequencyChannelAttention(
            in_channels=2048,
            dct_h=dct_h,
            dct_w=dct_w,
            frequency_branches=frequency_branches,
            frequency_selection=frequency_selection,
            reduction=reduction
        )
        self.rab = RAB(in_channels=2048, out_channels=2048, bias=True)
        self.mdcr = MDCR(in_features=2048, out_features=2048)
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.mfca(self.gap(self.rab(self.mdcr(x)))) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.mfca(self.gap(self.rab(self.mdcr(inputs))))
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_freq_indices(method):
    # 确保方法在指定的选项中
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    # 从方法名中提取频率数
    num_freq = int(method[3:])
    if 'top' in method:
        # 预定义的 top 频率索引
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        # 选择前 num_freq 个索引
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        # 预定义的 low 频率索引
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        # 选择前 num_freq 个索引
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        # 预定义的 bot 频率索引
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        # 选择前 num_freq 个索引
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        # 如果方法不在选项中，抛出异常
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h=7,
                 dct_w=7,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        # 确保频率分支数是有效的
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        # 构造频率选择字符串
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        # 获取频率索引
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        # 根据 DCT 大小调整索引
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # 确保 mapper_x 和 mapper_y 长度一致
        assert len(mapper_x) == len(mapper_y)

        # 初始化 DCT 权重
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        # 定义自适应池化层
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)
        self.model = TVConv(2048, h=1, w=1)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        # 初始化 DCT 滤波器
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        # 构建 DCT 滤波器
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        # 计算 DCT 滤波器值
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def forward(self, x):
        # 获取输入的形状
        batch_size, C, H, W = x.shape

        x_pooled = x
        # 如果输入大小与 DCT 大小不匹配，进行自适应池化
        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        # 初始化频谱特征
        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            # 循环遍历模型的状态字典，该字典包含模型的所有参数。它寻找名称中包含 'dct_weight' 的参数。
            if 'dct_weight' in name:
                # 计算频谱特征：将输入与 DCT 权重参数逐元素相乘
                x_pooled_spectral = x_pooled * params
                # 累加池化频谱特征的平均值
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                # 累加池化频谱特征的最大值
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                # 累加池化频谱特征的最小值：通过取反后最大池化的方法实现
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)


        # 归一化频谱特征
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq

        # 通过全连接层生成注意力图
        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        # 计算最终的注意力图
        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        # 将注意力图应用于输入
        x = x * multi_spectral_attention_map.expand_as(x)
        x=self.model(x)
        return x


import torch
import torch.nn as nn

class _ConvBlock(nn.Sequential):
    """
    _ConvBlock类定义了一个简单的卷积块，包含卷积层、层归一化和ReLU激活函数。
    """
    def __init__(self, in_planes, out_planes, h, w, kernel_size=3, stride=1, bias=False):
        # 计算填充大小，使得输出大小与输入大小相同
        padding = (kernel_size - 1) // 2
        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),  # 卷积层
            nn.LayerNorm([out_planes, h, w]),  # 层归一化
            nn.ReLU(inplace=True)  # ReLU激活函数
        )

class TVConv(nn.Module):
    """
    TVConv类定义了一个基于位置映射的空间变体卷积模块。
    """
    def __init__(self,
                 channels,
                 TVConv_k=3,
                 stride=1,
                 TVConv_posi_chans=4,
                 TVConv_inter_chans=64,
                 TVConv_inter_layers=3,
                 TVConv_Bias=False,
                 h=3,
                 w=3,
                 **kwargs):
        super(TVConv, self).__init__()

        # 注册缓冲区变量，表示卷积核大小、步长、通道数等
        self.register_buffer("TVConv_k", torch.as_tensor(TVConv_k))
        self.register_buffer("TVConv_k_square", torch.as_tensor(TVConv_k**2))
        self.register_buffer("stride", torch.as_tensor(stride))
        self.register_buffer("channels", torch.as_tensor(channels))
        self.register_buffer("h", torch.as_tensor(h))
        self.register_buffer("w", torch.as_tensor(w))

        self.bias_layers = None

        # 计算输出通道数
        out_chans = self.TVConv_k_square * self.channels

        # 初始化位置映射参数
        self.posi_map = nn.Parameter(torch.Tensor(1, TVConv_posi_chans, h, w))
        nn.init.ones_(self.posi_map)  # 用1初始化

        # 创建权重层和偏置层
        self.weight_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, out_chans, TVConv_inter_layers, h, w)
        if TVConv_Bias:
            self.bias_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, channels, TVConv_inter_layers, h, w)

        # 初始化 Unfold 模块，用于提取局部区域
        self.unfold = nn.Unfold(TVConv_k, 1, (TVConv_k-1)//2, stride)

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers, h, w):
        """
        创建卷积层序列。
        """
        layers = [_ConvBlock(in_chans, inter_chans, h, w, bias=False)]
        for i in range(num_inter_layers):
            layers.append(_ConvBlock(inter_chans, inter_chans, h, w, bias=False))
        layers.append(nn.Conv2d(
            in_channels=inter_chans,
            out_channels=out_chans,
            kernel_size=3,
            padding=1,
            bias=False))  # 最后一层卷积
        return nn.Sequential(*layers)

    def forward(self, x):
        # 计算卷积权重
        weight = self.weight_layers(self.posi_map)
        weight = weight.view(1, self.channels, self.TVConv_k_square, self.h, self.w) # torch.Size([1, 64, 9, 32, 32])
        # 利用 Unfold 模块获取局部区域，并按照权重进行加权求和
        out = self.unfold(x).view(x.shape[0], self.channels, self.TVConv_k_square, self.h, self.w) # torch.Size([2, 64, 9, 32, 32])
        """
            weight * out：对这两个张量在 TVConv_k_square 维度上进行逐元素相乘。这个操作相当于对每个位置的局部区域应用一个位置特定的卷积核。
            .sum(dim=2) ：在TVConv_k_square维度上对乘积结果进行求和。TVConv_k_square 代表卷积核的展开大小（即核的面积），
                所以这个求和操作相当于对每个局部区域的卷积结果进行加权求和，类似于传统卷积操作。        
        """
        out = (weight * out).sum(dim=2) #实现了基于位置的加权卷积操作，生成了一个新的特征图。 # torch.Size([2, 64, 32, 32])
        if self.bias_layers is not None:
            # 如果使用偏置，则加上偏置
            bias = self.bias_layers(self.posi_map)
            out = out + bias
        return out


from torch import nn
import math
import torch

class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()  # 初始化父类
        # 计算卷积核大小
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1  # 确保卷积核大小为奇数
        padding = kernel_size // 2  # 计算填充大小
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)  # 自适应平均池化到 1x1
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),  # 一维卷积
            nn.Sigmoid()  # Sigmoid 激活函数
        )

    def forward(self, x):
        out = self.pool(x)  # 对输入 x 进行自适应平均池化
        out = out.view(x.size(0), 1, x.size(1))  # 调整张量形状以适应卷积
        out = self.conv(out)  # 应用一维卷积和 Sigmoid
        out = out.view(x.size(0), x.size(1), 1, 1)  # 恢复张量形状

        return out * x  # 通道注意力乘以原始输入


import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups = groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class MDCR(nn.Module):
    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()

        self.block1 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.block2 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.block3 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.block4 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.out_s = conv_block(
            in_features=4,
            out_features=4,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )
        self.out = conv_block(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
            )
        self.eca_layer = ECA(in_channel=2048)

    def forward(self, x):
        split_tensors = []
        x = torch.chunk(x, 4, dim=1)
        x1 = self.block1(x[0])
        x2 = self.block2(x[1])
        x3 = self.block3(x[2])
        x4 = self.block4(x[3])
        for channel in range(x1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2, x3, x4]]
            concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))  # 拼接在 batch_size 维度上
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)
        x = self.out(x)
        x = self.eca_layer(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        # 可学习的权重参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # 可学习的偏置参数，初始化为全 0
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # 用于数值稳定性的小常数
        self.eps = eps
        # 数据格式，支持 "channels_last" 和 "channels_first"
        self.data_format = data_format
        # 检查数据格式是否合法，若不合法则抛出异常
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # 归一化的形状
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 如果数据格式为 "channels_last"
        if self.data_format == "channels_last":
            # 直接调用 PyTorch 的层归一化函数
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 如果数据格式为 "channels_first"
        elif self.data_format == "channels_first":
            # 计算通道维度上的均值
            u = x.mean(1, keepdim=True)
            # 计算通道维度上的方差
            s = (x - u).pow(2).mean(1, keepdim=True)
            # 进行归一化操作
            x = (x - u) / torch.sqrt(s + self.eps)
            # 应用可学习的权重和偏置
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Intensity Enhancement Layer，强度增强层
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        # 调用父类的构造函数
        super(IEL, self).__init__()
        # 计算隐藏层的特征维度
        hidden_features = int(dim * ffn_expansion_factor)
        # 输入投影层，将输入特征维度映射到隐藏层特征维度的 2 倍
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 深度可分离卷积层 1
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        # 深度可分离卷积层 2
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        # 深度可分离卷积层 3
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        # 输出投影层，将隐藏层特征维度映射回输入特征维度
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        # Tanh 激活函数
        self.Tanh = nn.Tanh()

    def forward(self, x):
        # 输入投影
        x = self.project_in(x)
        # 将特征图在通道维度上拆分为两部分
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 对 x1 应用深度可分离卷积和 Tanh 激活函数，并加上残差连接
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        # 对 x2 应用深度可分离卷积和 Tanh 激活函数，并加上残差连接
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        # 将 x1 和 x2 逐元素相乘
        x = x1 * x2
        # 输出投影
        x = self.project_out(x)
        return x

# Cross Attention Block，交叉注意力块
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        # 调用父类的构造函数
        super(CAB, self).__init__()
        # 注意力头的数量
        self.num_heads = num_heads
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 查询卷积层
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 查询的深度可分离卷积层
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 键值卷积层
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        # 键值的深度可分离卷积层
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        # 输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # 获取输入特征图的形状
        b, c, h, w = x.shape
        # 计算查询
        q = self.q_dwconv(self.q(x))

        # 计算键值
        kv = self.kv_dwconv(self.kv(y))
        # 将键值在通道维度上拆分为键和值
        k, v = kv.chunk(2, dim=1)

        # 对查询、键和值进行维度重排
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对查询和键进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 对注意力分数进行 softmax 操作
        attn = nn.functional.softmax(attn, dim=-1)
        # 计算注意力输出
        out = (attn @ v)
        # 对注意力输出进行维度重排
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 输出投影
        out = self.project_out(out)
        return out

# Lightweight Cross Attention，轻量级交叉注意力
class LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        # 调用父类的构造函数
        super(LCA, self).__init__()
        # 强度增强层
        self.gdfn = IEL(dim)
        # 层归一化层
        self.norm = LayerNorm(dim)
        # 交叉注意力块
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x)) + x
        return x


import torch
import torch.nn as nn

class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        # 调用父类的构造函数
        super(Basic, self).__init__()
        # 输出通道数
        self.out_channels = out_planes
        # 卷积层的组数，默认为1
        groups = 1
        # 定义二维卷积层
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 通过卷积层
        x = self.conv(x)
        # 通过ReLU激活函数
        x = self.relu(x)
        # 返回结果
        return x

class ChannelPool(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(ChannelPool, self).__init__()

    def forward(self, x):
        # 在通道维度上取最大值和平均值，并在新的维度上拼接
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(SAB, self).__init__()
        # 定义卷积核大小
        kernel_size = 5
        # 定义通道池化操作
        self.compress = ChannelPool()
        # 定义空间注意力机制中的卷积操作
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        # 压缩通道信息
        x_compress = self.compress(x)
        # 通过空间卷积
        x_out = self.spatial(x_compress)
        # 通过Sigmoid函数生成注意力图
        scale = torch.sigmoid(x_out)
        # 返回加权后的特征图
        return x * scale


class RAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        # 调用父类的构造函数
        super(RAB, self).__init__()
        # 定义卷积核大小
        kernel_size = 3
        # 定义步长
        stride = 1
        # 定义填充
        padding = 1
        # 初始化卷积层列表
        layers = []
        # 添加第一个卷积层和ReLU激活
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        # 添加第二个卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        # 将卷积层组成一个序列
        self.res = nn.Sequential(*layers)
        # 定义空间注意力模块
        self.sab = SAB()

        self.module = LCA(dim=2048, num_heads=8)
    def forward(self, x):
        # 第一次残差连接
        x1 = x + self.res(x)
        # 第二次残差连接
        x2 = x1 + self.res(x1)
        # 第三次残差连接
        x3 = x2 + self.res(x2)

        # 叠加x1和x3
        x3_1 = x1 + x3
        # 第四次残差连接
        x4 = x3_1 + self.res(x3_1)
        # 叠加x和x4
        x4_1 = x + x4

        # 通过空间注意力模块
        x5 = self.sab(x4_1)
        # 叠加x和x5
        x5_1 = self.module(x,x5)

        # 返回最终结果
        return x5_1