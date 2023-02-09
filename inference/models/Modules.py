import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter
import mindspore.ops.operations as P
import mindspore.common.initializer as weight_init
from mindspore import Tensor, ops
from mindspore import dtype as mstype


class SiLU(nn.Cell):
    """SiLU"""

    def __init__(self):
        super(SiLU, self).__init__()
        self.ops_sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.ops_sigmoid(x)

    def __repr__(self):
        return "SiLU<x * Sigmoid(x)>"


class SELayer(nn.Cell):
    """SELayer"""

    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.fc = nn.SequentialCell([
            nn.Conv2d(in_channels=oup, out_channels=inp // reduction,
                      kernel_size=1, has_bias=True),
            SiLU(),
            nn.Conv2d(in_channels=inp // reduction, out_channels=oup,
                      kernel_size=1, has_bias=True),
            nn.Sigmoid()
        ])

    def construct(self, x):
        y = self.avg_pool(x, [2, 3])
        y = self.fc(y)
        return y * x


def conv_3x3_bn(inp, oup, stride, norm_type):
    """conv_3x3_bn"""
    return nn.SequentialCell([
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, pad_mode='same', has_bias=False),
        norm_type(num_features=oup, momentum=0.9, eps=1e-3),
        SiLU()
    ])


def conv_1x1_bn(inp, oup, norm_type):
    """conv_1x1_bn"""
    return nn.SequentialCell([
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, has_bias=False),
        norm_type(num_features=oup, momentum=0.9, eps=1e-3),
        SiLU()
    ])


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks). """

    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath2D(DropPath):
    """DropPath2D"""

    def __init__(self, drop_prob):
        super(DropPath2D, self).__init__(drop_prob=drop_prob, ndim=2)


class ConvBnrelu2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size), stride=(stride, stride), pad_mode='pad',
                              padding=padding, dilation=(dilation, dilation), group=groups, has_bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9, eps=1e-3)
        self.relu = SiLU()
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, (nn.Dense, nn.Conv2d)) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def construct(self, x):
        return self.relu(self.bn(self.conv(x)))


class RepConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(RepConv, self).__init__()
        self.conv13 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(13, 13), stride=(1, 1),
                                pad_mode='pad', padding=6)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                               pad_mode='pad', padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=0.9, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=0.9, eps=1e-3)
        self.relu = SiLU()
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, (nn.Dense, nn.Conv2d)) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def construct(self, x):
        return self.relu(self.bn1(self.conv13(x))+self.bn2(self.conv3(x)))


class ASPP(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.conv1 = ConvBnrelu2d(in_channels, out_channels)
        self.Dconv1 = ConvBnrelu2d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1)

        self.conv2 = ConvBnrelu2d(in_channels, out_channels)
        self.Dconv2 = ConvBnrelu2d(out_channels, out_channels, kernel_size=3, dilation=6, padding=6)

        self.conv3 = ConvBnrelu2d(in_channels, out_channels)
        self.Dconv3 = ConvBnrelu2d(out_channels, out_channels, kernel_size=3, dilation=12, padding=12)

        self.conv4 = ConvBnrelu2d(in_channels, out_channels)
        self.Dconv4 = ConvBnrelu2d(out_channels, out_channels, kernel_size=3, dilation=18, padding=18)

        self.conv5 = ConvBnrelu2d(out_channels * 4, in_channels)

    def construct(self, x):
        x_0 = x
        x_1 = self.Dconv1(self.conv1(x))
        x_2 = self.Dconv2(self.conv2(x))
        x_3 = self.Dconv3(self.conv3(x))
        x_4 = self.Dconv4(self.conv4(x))
        x = P.Concat(1)((x_1, x_2, x_3, x_4))
        x = self.conv5(x)
        x = x + x_0
        return x


class ASPP_Global(nn.Cell):
    def __init__(self, in_channels, hide_channels):
        super(ASPP_Global, self).__init__()

        self.conv6 = ConvBnrelu2d(in_channels, in_channels)

    def construct(self, x):
        b, c, h, w = x.shape
        x_0 = x

        x = P.Reshape()(x, (b, c, h * w))
        attention1 = x
        attention1 = attention1 / (nn.Norm(axis=1, keep_dims=True)(attention1) + 1e-5)
        attention1 = P.ReLU()(ops.BatchMatMul()(attention1, ops.Transpose()(attention1, (0, 2, 1))))
        attention1 = attention1 / (ops.ReduceSum(keep_dims=True)(attention1, 1) + 1e-5)
        x_5 = P.Reshape()(ops.BatchMatMul()(attention1, x), (b, -1, h, w))

        attention2 = P.Reshape()(x_0, (b, c, h * w))
        attention2 = attention2 / (nn.Norm(axis=1, keep_dims=True)(attention2) + 1e-5)
        attention2 = P.ReLU()(ops.BatchMatMul()(attention2, ops.Transpose()(attention2, (0, 2, 1))))
        attention2 = attention2 / (ops.ReduceSum(keep_dims=True)(attention2, 1) + 1e-5)
        x_6 = P.Reshape()(ops.BatchMatMul()(attention2, x), (b, -1, h, w))

        x = self.conv6(x_0 + x_5 + x_6)
        return x


class Spatial_ASPP(nn.Cell):
    def __init__(self, in_channels, hide_channels):
        super(Spatial_ASPP, self).__init__()

    def construct(self, x):
        b, c, h, w = x.shape
        x_0 = x

        x = P.Reshape()(x, (b, c, h * w))
        attention1 = x
        attention2 = P.Reshape()(x_0, (b, c, h * w))

        attention1 = attention1 / (nn.Norm(axis=1, keep_dims=True)(attention1) + 1e-5)
        attention1 = P.ReLU()(ops.BatchMatMul()(ops.Transpose()(attention1, (0, 2, 1)), attention1))
        attention1 = attention1 / (ops.ReduceSum(keep_dims=True)(attention1, 1) + 1e-5)
        x_5 = P.Reshape()(ops.BatchMatMul()(x, attention1), (b, -1, h, w))

        attention2 = attention2 / (nn.Norm(axis=1, keep_dims=True)(attention2) + 1e-5)
        attention2 = P.ReLU()(ops.BatchMatMul()(ops.Transpose()(attention2, (0, 2, 1)), attention2))
        attention2 = attention2 / (ops.ReduceSum(keep_dims=True)(attention2, 1) + 1e-5)
        x_6 = P.Reshape()(ops.BatchMatMul()(x, attention2), (b, -1, h, w))

        x = x_0 + x_5 + x_6
        return x


class DeformConv2d(nn.Cell):
    def __init__(self, inc, outc, kernel_size=5, stride=1, p_size=12):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kernel_size, stride=kernel_size,
                              pad_mode='pad', has_bias=True)

        self.p_conv = nn.Conv2d(in_channels=inc, out_channels=2 * kernel_size * kernel_size, kernel_size=5,
                                stride=stride, pad_mode='pad', padding=2, has_bias=True)

        p_n_g1 = Tensor(np.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1, 1))
        self.p_n_g1 = Parameter(p_n_g1, requires_grad=False)
        p_n_g2 = Tensor(np.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1, 1))
        self.p_n_g2 = Parameter(p_n_g2, requires_grad=False)

        p_0_g1 = Tensor(np.arange(1, p_size * self.stride + 1, self.stride))
        self.p_0_g1 = Parameter(p_0_g1, requires_grad=False)
        p_0_g2 = Tensor(np.arange(1, p_size * self.stride + 1, self.stride))
        self.p_0_g2 = Parameter(p_0_g2, requires_grad=False)

    def construct(self, x):
        offset = self.p_conv(x)

        dtype = offset.dtype
        ks = self.kernel_size
        N = P.Shape()(offset)[1] // 2  # k**2

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = P.Transpose()(p, (0, 2, 3, 1))
        q_lt = ops.Floor()(ops.stop_gradient(p))
        q_rb = q_lt + 1

        q_lt = P.Concat(-1)([ops.clip_by_value(q_lt[..., :N], 0, P.Shape()(x)[2] - 1),
                             ops.clip_by_value(q_lt[..., N:], 0, P.Shape()(x)[3] - 1)]).astype(mstype.int32)
        q_rb = P.Concat(-1)([ops.clip_by_value(q_rb[..., :N], 0, P.Shape()(x)[2] - 1),
                             ops.clip_by_value(q_rb[..., N:], 0, P.Shape()(x)[3] - 1)]).astype(mstype.int32)
        q_lb = P.Concat(-1)([q_lt[..., :N], q_rb[..., N:]])
        q_rt = P.Concat(-1)([q_rb[..., :N], q_lt[..., N:]])

        # clip p
        p = P.Concat(-1)([ops.clip_by_value(p[..., :N], 0, P.Shape()(x)[2] - 1),
                          ops.clip_by_value(p[..., N:], 0, P.Shape()(x)[3] - 1)])

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].astype(p.dtype) - p[..., :N])) * (1 + (q_lt[..., N:].astype(p.dtype) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].astype(p.dtype) - p[..., :N])) * (1 - (q_rb[..., N:].astype(p.dtype) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].astype(p.dtype) - p[..., :N])) * (1 - (q_lb[..., N:].astype(p.dtype) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].astype(p.dtype) - p[..., :N])) * (1 + (q_rt[..., N:].astype(p.dtype) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = ops.ExpandDims()(g_lt, 1) * x_q_lt + \
                   ops.ExpandDims()(g_rb, 1) * x_q_rb + \
                   ops.ExpandDims()(g_lb, 1) * x_q_lb + \
                   ops.ExpandDims()(g_rt, 1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    # def _get_p_n(self, N, dtype):
    #     p_n_x, p_n_y = ms.numpy.meshgrid(
    #         ms.numpy.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
    #         ms.numpy.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), indexing='ij')
    #     p_n = ms.ops.Reshape()(P.Concat(0)([p_n_x.flatten(), p_n_y.flatten()]), (1, 2*N, 1, 1)).astype(dtype)
    #     return p_n
    #
    # def _get_p_0(self, h, w, N, dtype):
    #     p_0_x, p_0_y = ms.numpy.meshgrid(
    #         ms.numpy.arange(1, h * self.stride + 1, self.stride),
    #         ms.numpy.arange(1, w * self.stride + 1, self.stride), indexing='ij')
    #     p_0_x = ms.ops.repeat_elements(ms.ops.Reshape()(p_0_x, (1, 1, h, w)), N, axis=1)
    #     p_0_y = ms.ops.repeat_elements(ms.ops.Reshape()(p_0_y, (1, 1, h, w)), N, axis=1)
    #     p_0 = P.Concat(1)([p_0_x, p_0_y]).astype(dtype)
    #     return p_0

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = ms.ops.Meshgrid(indexing='ij')((self.p_n_g1, self.p_n_g2))
        p_n = ops.Reshape()(P.Concat(0)([p_n_x.flatten(), p_n_y.flatten()]), (1, 2*N, 1, 1)).astype(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = ms.ops.Meshgrid(indexing='ij')((self.p_0_g1, self.p_0_g2))
        p_0_x = ops.repeat_elements(ops.Reshape()(p_0_x, (1, 1, h, w)), N, axis=1)
        p_0_y = ops.repeat_elements(ops.Reshape()(p_0_y, (1, 1, h, w)), N, axis=1)
        p_0 = P.Concat(1)([p_0_x, p_0_y]).astype(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = P.Shape()(offset)[1] // 2, P.Shape()(offset)[2], P.Shape()(offset)[3]
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = P.Shape()(q)
        padded_w = P.Shape()(x)[3]
        c = P.Shape()(x)[1]
        # (b, c, h*w)
        x = x.view(b, c, -1)
        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = ops.BroadcastTo((-1, c, -1, -1, -1))(ops.ExpandDims()(index, 1)).view(b, c, -1)
        x_offset = ops.GatherD()(x, -1, index).view(b, c, h, w, N)
        return x_offset

    def _reshape_x_offset(self, x_offset, ks):
        b, c, h, w, N = P.Shape()(x_offset)
        x_offset = x_offset.view(b, c, h, w * ks, ks)
        x_offset = x_offset.view(b, c, h * ks, w * ks)
        return x_offset


if __name__ == "__main__":
    from mindspore import context

    dcn = DeformConv2d(3, 12)
    x = ops.StandardNormal(2)((2, 3, 120, 160))
    # context.set_context(mode=context.GRAPH_MODE)
    context.set_context(mode=context.PYNATIVE_MODE)
    print(x.shape)
    y = dcn(x)
    print(y.shape)
