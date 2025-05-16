from unittest.mock import inplace

import torch
import torch.nn as nn
from tensorflow.python.layers.core import dropout

from utils import InitWeights_He
from torchvision import ops
import torch.nn.functional as F
import  numpy as np
import math

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class convblock(nn.Module):
    def __init__(self, in_c, out_c, dp=0, pooling=False):
        super(convblock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_c, out_c, kernel_size=3, padding=1, bias=False),
            #nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            # DepthwiseSeparableConv(out_c, out_c, kernel_size=3, padding=1, bias=False),
            # #nn.InstanceNorm2d(out_c),
            # nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_c, out_c, kernel_size=3, padding=1, bias=False),
            # nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            DepthwiseSeparableConv(out_c, out_c, kernel_size=3, padding=1, bias=False),
            # nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp))
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.pooling = pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.pooling:
            x = self.pool(x)
        identity = self.conv11(x)
        out = self.conv(x)
        out = out + identity
        return self.relu(out)

class left_conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0, pooling=False,kz=3,pd=1):
        super(left_conv, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='lad', in_channels=in_c, out_channels=out_c, kernel_size=kz, padding=pd),
                                    nn.Dropout2d(dp),
                                    # nn.ReLU(inplace=True),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    DepthwiseSeparableConv(out_c, out_c, kernel_size=kz, padding=pd, bias=False),
                                    nn.Dropout2d(dp)])
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.pooling = pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        if self.pooling:
            x = self.pool(x)
        identity = self.conv11(x)
        out = self.conv(x)
        out = out + identity
        return self.relu(out)

class right_conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0, pooling=False,kz=3,pd=1):
        super(right_conv, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='rad', in_channels=in_c, out_channels=out_c, kernel_size=kz, padding=pd),
                                    nn.Dropout2d(dp),
                                    # nn.ReLU(inplace=True),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    DepthwiseSeparableConv(out_c, out_c, kernel_size=kz, padding=pd, bias=False),
                                    nn.Dropout2d(dp)])
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.pooling = pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.pooling:
            x = self.pool(x)
        identity = self.conv11(x)
        out = self.conv(x)
        out = out + identity
        return self.relu(out)

class BasicDeformConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, bias=False):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size
        mask_channels = kernel_size * kernel_size
        self.def_conv2d = ops.DeformConv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2d_offset = nn.Conv2d(in_c, offset_channels, kernel_size=kernel_size, padding=padding, bias=bias)
       # nn.init.constant_(self.conv2d_offset.weight.data, 0)
        self.conv2d_mask = nn.Conv2d(in_c, mask_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        #nn.init.constant_(self.conv2d_mask.weight.data, 0.5)

    def forward(self, x):
        offset = self.conv2d_offset(x)
        mask = torch.sigmoid(self.conv2d_mask(x))
        out = self.def_conv2d(x,offset,mask)
        return out

class deformableblock(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(deformableblock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.deform_conv = nn.Sequential(
            BasicDeformConv2d(in_c, out_c//2, kernel_size=3, padding=1, bias=False),
            #nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            BasicDeformConv2d( out_c//2, out_c, kernel_size=3, padding=1, bias=False),
            # nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            BasicDeformConv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            # nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp))
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        identity = self.conv11(x)
        out = self.deform_conv(x)
        out = out + identity
        return self.relu(out)

class dilateblock(nn.Module):
    def __init__(self, in_c, out_c, dp=0, pooling=False):
        super(dilateblock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.dilate_conv = nn.Sequential(
            DepthwiseSeparableConv(in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2),
            #nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            # DepthwiseSeparableConv(out_c, out_c, kernel_size=3, padding=5, bias=False, dilation=5),
            # #nn.InstanceNorm2d(out_c),
            # nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2),
            # nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            DepthwiseSeparableConv(out_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2),
            # nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp))
        self.conv11 =  nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling = pooling
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.pooling:
            x = self.pool(x)
        identity = self.conv11(x)
        out = self.dilate_conv(x)
        out = out + identity
        return self.relu(out)

class inhibitionblock(nn.Module):
    def __init__(self, in_c, out_c, dp=0, pooling=False):
        super(inhibitionblock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.pooling = pooling
        self.CRF = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False, dilation=1),
            #nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            DepthwiseSeparableConv(out_c, out_c, kernel_size=3, padding=1, bias=False, dilation=1),
            #nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True))
        self.NCRF1 = nn.Sequential(
            DepthwiseSeparableConv(in_c, out_c, kernel_size=3, padding=1, bias=False, dilation=1),
            #nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            DepthwiseSeparableConv(out_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2),
            #nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True))
        # self.NCRF2 = nn.Sequential(
        #     DepthwiseSeparableConv(in_c, out_c, kernel_size=5, padding=2, bias=False, dilation=1),
        #     # nn.InstanceNorm2d(out_c),
        #     nn.Dropout2d(dp),
        #     # nn.ReLU(inplace=True),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     DepthwiseSeparableConv(out_c, out_c, kernel_size=5, padding=4, bias=False, dilation=2),
        #     # nn.InstanceNorm2d(out_c),
        #     nn.Dropout2d(dp),
        #     # nn.ReLU(inplace=True),
        #     nn.LeakyReLU(0.1, inplace=True))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False)
        #self.conv1_1 = nn.Conv2d(2*out_c, out_c, kernel_size=1, padding=0, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.pooling:
            x = self.pool(x)
        identity = self.conv11(x)
        CRFout = self.CRF(x)
        NCRFout1 = self.NCRF1(x)
        # NCRFout2 = self.NCRF2(x)
        # out = CRFout + CRFout - NCRFout1 - NCRFout2
        out = CRFout - NCRFout1
        out = out + identity
        #out = torch.cat([out1, out2],dim=1)
        #out = self.conv1_1(out) + identity
        return self.relu(out)

class adap_conv1(nn.Module):
    def __init__(self, in_channels, out_channels,kz=3,pd=1):
        super(adap_conv1, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='2rd', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    # nn.ReLU(inplace=True),
                                    nn.LeakyReLU(0.1, inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x

class adap_conv2(nn.Module):
    def __init__(self, in_channels, out_channels,kz=3,pd=1):
        super(adap_conv2, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='2rd', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    # nn.ReLU(inplace=True),
                                    nn.LeakyReLU(0.1, inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x

class fusionblock(nn.Module):
    def __init__(self, in_c, out_c, dp=0, up= False):
        super(fusionblock, self).__init__()
        self.pre_conv1 = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.ReLU(inplace=True))
        self.pre_conv2 = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.ReLU(inplace=True))
        # self.pre_conv1 = adap_conv1(in_c[0], out_c,kz=3,pd=1)
        # self.pre_conv2 = adap_conv2(in_c[1], out_c,kz=3,pd=1)
        self.up =up
        self.deconv_weight = nn.Parameter(bilinear_upsample_weights(2, out_c))
        # self.conv1_1 = nn.Conv2d(in_c[0], out_c, kernel_size=1)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, *x):
        x1 = self.pre_conv1(x[0])
        x2 = self.pre_conv2(x[1])
        if self.up:
            x1 = F.conv_transpose2d(x1, self.deconv_weight, padding=1, stride=2,
                                     output_padding=(x2.size(2) - x1.size(2) * 2, x2.size(3) - x1.size(3) * 2))
        #out = torch.cat([x1, x2],dim=1)

        out = x1 + x2
        return self.relu(out)


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((number_of_classes,
                        number_of_classes,
                        filter_size,
                        filter_size,), dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[i, i, :, :] = upsample_kernel
    return torch.Tensor(weights)

def createPDCFunc(PDC_type):  # 创建像素差卷积函数
    assert PDC_type in ['cv', 'lad', 'rad', '2sd','2rd', '2xd'], 'unknown PDC type: %s' % str(PDC_type)

    if PDC_type == 'cv':  # 采用香草卷积
        return F.conv2d
    if PDC_type == 'rad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func

    elif PDC_type == 'lad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [1, 2, 5, 0, 4, 8, 3, 6, 7]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif PDC_type == '2sd':  # CPDC,基于周围差的像素差卷积
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 or weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            # assert padding == dilation, 'padding for ad_conv set wrong'
            # print('0', weights[0][0])
            shape = weights.shape
            # if weights.is_cuda:
            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)
            # else:
            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            # print(buffer)
            # buffer = weights
            # 1 2 3
            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]
            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] = buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] + buffer[:, :,
                                                                                              [2, 7, 8, 5, 3, 0, 1,
                                                                                               6]] - 2 * buffer[:, :,
                                                                                                         [1, 4, 5, 4, 4,
                                                                                                          3, 4, 7]]
            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]
            buffer[:, :, [4]] = 0
            weights = buffer.view(shape)
            # print(weights[0][0])
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    elif PDC_type == '2rd':  # CPDC,基于周围差的像素差卷积
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 or weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            # assert padding == dilation, 'padding for ad_conv set wrong'
            # print('0', weights[0][0])
            shape = weights.shape
            # if weights.is_cuda:
            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)
            # else:
            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            # print(buffer)
            # buffer = weights
            # 1 2 3
            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]
            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] = buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] + buffer[:, :,
                                                                                              [6, 7, 0, 5, 3, 8, 1,
                                                                                               2]] - 2 * buffer[:, :,
                                                                                                         [3, 4, 2, 4, 4,
                                                                                                          7, 4, 5]]
            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]
            buffer[:, :, [4]] = 0
            weights = buffer.view(shape)
            # print(weights[0][0])
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    elif PDC_type == '2xd':  # CPDC,基于周围差的像素差卷积
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 or weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            # assert padding == dilation, 'padding for ad_conv set wrong'
            # print('0', weights[0][0])
            shape = weights.shape
            # if weights.is_cuda:
            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)
            # else:
            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            # print(buffer)
            # buffer = weights
            # 1 2 3
            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]
            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] = buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] + buffer[:, :,
                                                                                              [8, 7, 6, 5, 3, 2, 1,
                                                                                               0]] - 2 * buffer[:, :,
                                                                                                         [4, 4, 4, 4, 4,
                                                                                                          4, 4, 4]]
            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]
            buffer[:, :, [4]] = 0
            weights = buffer.view(shape)
            # print(weights[0][0])
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    else:
        print('unknown PDC type: %s' % str(PDC_type))  # 正常来说走不到这里
        return None

class Conv2d(nn.Module):  # 把之前创建的卷积函数包装成torch卷积api相同的格式
    def __init__(self, pdc_func, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        """
        :param pdc_func: 卷积函数
        """
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:  # depth wise卷积要求通道要能被分组数整除
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation  # 用于控制空洞卷积，默认为1，卷积核尺寸不膨胀
        self.groups = groups
        # print(self.kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc_func = createPDCFunc(pdc_func)

    def reset_parameters(self):
        # 凯明初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        #               输入      卷积核权重     偏置                                卷积核膨胀（用于空洞卷积）
        return self.pdc_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PVP_UNet_tiny(nn.Module):
    def __init__(self,  num_classes=1, num_channels=3, feature_scale=2,  dropout=0):
        super(PVP_UNet_tiny, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        # self.conv1 = DepthwiseSeparableConv(num_channels, filters[0], kernel_size=7, padding=6, dilation=2)
        # self.relu = nn.ReLU(inplace=True)
        self.deformconv = deformableblock(num_channels, filters[1],dp=dropout)
        # self.right_retinaconv = right_conv(filters[0],filters[1],dp=dropout)
        # self.left_retinaconv = left_conv(filters[0], filters[1], dp=dropout)
        self.LGNdilateconv = dilateblock(filters[1],filters[1],dp=dropout, pooling=True)
        self.LGNnormconv = convblock(filters[1],filters[1],dp=dropout, pooling=True)
        self.inhibitconv1 = inhibitionblock(filters[1], filters[1], dp=dropout, pooling=True)
        self.inhibitconv2 = inhibitionblock(filters[1], filters[2], dp=dropout, pooling=True)
        self.inhibitconv3 = inhibitionblock(filters[2], filters[2], dp=dropout, pooling=True)
        self.inhibitconv4 = inhibitionblock(filters[2], filters[2], dp=dropout)
        self.fusionconv1 = fusionblock(in_c=(filters[2],filters[2]),out_c=filters[2],dp=dropout)
        self.fusionconv2 = fusionblock(in_c=(filters[2],filters[2]),out_c=filters[1],dp=dropout, up=True)
        self.fusionconv3 = fusionblock(in_c=(filters[1], filters[1]), out_c=filters[1], dp=dropout, up=True)
        self.fusionconv4 = fusionblock(in_c=(filters[1],filters[1]),out_c=filters[1],dp=dropout, up=True)
        self.fusionconv5 = fusionblock(in_c=(filters[1], filters[1]), out_c=filters[0], dp=dropout, up=True)
        self.fuse = nn.Conv2d(filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.apply(InitWeights_He)

    def forward(self, x):
        num_chunks = 2
        # temp = self.relu(self.conv1(x))
        # temp = self.deformconv(x)
        retina_l = self.deformconv(x)
        retina_r =self.deformconv(x)

        separated_l = torch.chunk(retina_l, num_chunks, dim=1)
        separated_r = torch.chunk(retina_r, num_chunks, dim=1)
        LGN_l = torch.cat([separated_l[0], separated_r[0]],dim=1)
        LGN_r = torch.cat([separated_l[1], separated_r[1]],dim=1)
        dilateconv_l = self.LGNdilateconv(LGN_l)
        dilateconv_r = self.LGNdilateconv(LGN_r)
        normconv_l = self.LGNnormconv(LGN_l)
        normconv_r = self.LGNnormconv(LGN_r)
        # catdilate = torch.cat([dilateconv_l,dilateconv_r], dim=1)
        # catnorm = torch.cat([ normconv_l, normconv_r], dim=1)
        # catdilnorm1 = torch.cat([dilateconv_l, normconv_r], dim=1)
        # catdilnorm2 = torch.cat([dilateconv_r, normconv_l], dim=1)
        inhibit_11 = self.inhibitconv1(dilateconv_l)
        inhibit_13 = self.inhibitconv1(dilateconv_r)
        inhibit_12 = self.inhibitconv1(normconv_l)
        inhibit_14 = self.inhibitconv1(normconv_r)
        inhibit_21 = self.inhibitconv2(inhibit_11 + inhibit_12)
        inhibit_22 = self.inhibitconv2(inhibit_12 + inhibit_13)
        inhibit_23 = self.inhibitconv2(inhibit_13 + inhibit_14)
        inhibit_31 = self.inhibitconv3(inhibit_21 + inhibit_22)
        inhibit_32 = self.inhibitconv3(inhibit_22 + inhibit_23)
        inhibit3 = inhibit_31 + inhibit_32

        inhibit2 = inhibit_21 + inhibit_22 + inhibit_23
        inhibit4 = self.inhibitconv4(inhibit3)
        inhibit1 = inhibit_11 + inhibit_12 + inhibit_13 + + inhibit_14
        add_LGN = dilateconv_l + dilateconv_r + normconv_l + normconv_r
        add_retina = retina_l + retina_r
        fusion_1 = self.fusionconv1(inhibit4, inhibit3)
        fusion_2 = self.fusionconv2(fusion_1, inhibit2)
        fusion_3 = self.fusionconv3(fusion_2, inhibit1)
        fusion_4 = self.fusionconv4(fusion_3, add_LGN)
        fusion_5 = self.fusionconv5(fusion_4, add_retina)
        out_put = self.fuse(fusion_5)
        return out_put.sigmoid()
