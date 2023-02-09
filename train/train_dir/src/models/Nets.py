import numpy as np
from collections import OrderedDict

import mindspore.nn as nn
from mindspore import ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .Modules import DropPath2D


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     pad_mode='pad', padding=padding, dilation=dilation, group=groups, has_bias=bias,
                     weight_init="XavierUniform")


def get_bn(channels):
    return nn.BatchNorm2d(num_features=channels, momentum=0.9)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = OrderedDict()
    result["conv"] = get_conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=False)
    result["bn"] = get_bn(out_channels)
    result = nn.SequentialCell(result)
    return result


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.append(nn.ReLU())
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.moving_mean
    running_var = bn.moving_variance
    gamma = bn.gamma
    beta = bn.beta
    eps = bn.eps
    std = ops.Sqrt()(running_var + eps)
    t = ops.Reshape()(gamma / std, (-1, 1, 1, 1))
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=1, groups=groups)
        self.lkb_reparam = None
        self.small_conv = None
        if small_kernel is not None:
            self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                      stride=stride, padding=small_kernel // 2, groups=groups, dilation=1)

    def construct(self, inputs):
        if self.lkb_reparam:
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if self.small_conv:
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin[0], self.lkb_origin[1])
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv[0], self.small_conv[1])
            eq_b += small_b
            #   add to the central part
            eq_k += nn.Pad(((0, 0), (0, 0), ((self.kernel_size - self.small_kernel) // 2,) * 2,
                            ((self.kernel_size - self.small_kernel) // 2,) * 2))(small_k)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin[0].in_channels,
                                      out_channels=self.lkb_origin[0].out_channels,
                                      kernel_size=self.lkb_origin[0].kernel_size, stride=self.lkb_origin[0].stride,
                                      padding=self.lkb_origin[0].padding, dilation=self.lkb_origin[0].dilation,
                                      groups=self.lkb_origin[0].group, bias=True)
        self.lkb_reparam.weight.set_data(eq_k)
        self.lkb_reparam.bias.set_data(eq_b)
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ConvFFN(nn.Cell):
    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath2D(drop_path) if drop_path > 0. else ops.Identity()
        self.preffn_bn = get_bn(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0,
                           groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                           groups=1)
        self.nonlinear = nn.GELU()

    def construct(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKBlock(nn.Cell):
    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels,
                                                   kernel_size=block_lk_size,
                                                   stride=1, groups=dw_channels, small_kernel=small_kernel)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = get_bn(in_channels)
        self.drop_path = DropPath2D(drop_path) if drop_path > 0. else ops.Identity()

    def construct(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKNetStage(nn.Cell):
    def __init__(self, channels, num_blocks, stage_lk_size, drop_path, small_kernel, dw_ratio=1., ffn_ratio=4):
        super().__init__()
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            replk_block = RepLKBlock(in_channels=channels, dw_channels=int(channels * dw_ratio),
                                     block_lk_size=stage_lk_size,
                                     small_kernel=small_kernel, drop_path=block_drop_path)
            convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio),
                                    out_channels=channels,
                                    drop_path=block_drop_path)
            blks.append(replk_block)
            blks.append(convffn_block)
        self.blocks = nn.SequentialCell(blks)
        self.norm = ops.Identity()

    def construct(self, x):
        x = self.blocks(x)
        return x


class RepLKNet(nn.Cell):
    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
                 dw_ratio=1., ffn_ratio=4, in_channels=3, num_classes=1000, drop_rate=0.):
        super().__init__()
        base_width = channels[0]
        self.num_stages = len(layers)
        self.stem = nn.SequentialCell([
            conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1,
                         groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1,
                         groups=base_width),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1,
                         groups=base_width)])

        dpr = [r.item() for r in np.linspace(0, drop_path_rate, sum(layers))]
        self.stages = nn.CellList()
        self.transitions = nn.CellList()
        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.SequentialCell([
                    conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1,
                                 groups=channels[stage_idx + 1])])
                self.transitions.append(transition)

        if num_classes is not None:
            self.norm = get_bn(channels[-1])
            self.avgpool = ops.ReduceMean(keep_dims=False)
            self.head = nn.Dense(in_channels=channels[-1], out_channels=num_classes)

        self.dropout = nn.Dropout(keep_prob=1 - drop_rate)

    def forward_features(self, x):
        x = self.stem(x)

        for stage_idx in range(self.num_stages):
            x = self.stages[stage_idx](x)
            if stage_idx < self.num_stages - 1:
                x = self.transitions[stage_idx](x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = self.avgpool(x, [2, 3])
        x = self.dropout(x)
        x = self.head(x)
        return x

    def structural_reparam(self):
        for _, cell in self.cells_and_names():
            if hasattr(cell, 'merge_kernel'):
                cell.merge_kernel()


def create_RepLKNet31XL(args, pretrained=False):
    drop_path_rate = args.drop_path_rate
    num_classes = args.num_classes
    drop_rate = args.drop_rate
    model = RepLKNet(large_kernel_sizes=[27, 27, 27, 13], layers=[2, 2, 18, 2], channels=[256, 512, 1024, 2048],
                     drop_path_rate=drop_path_rate, small_kernel=None, num_classes=num_classes, drop_rate=drop_rate,
                     dw_ratio=1.5)
    if pretrained and args.pretrain_path:
        params_dict = {}
        for name, param in model.parameters_and_names():
            params_dict[name] = param.name
        print('Begin load pretrained model!')
        param_dict = load_checkpoint(args.pretrain_path + 'weight_xl.ckpt')
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != 256:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        new_param_dict = {}
        for key, value in param_dict.copy().items():
            new_param_dict[params_dict[key]] = value
        load_param_into_net(model, new_param_dict)
        print('Load pretrained model success!')
    return model


class MSNet(nn.Cell):
    def __init__(self, args):
        super(MSNet, self).__init__()
        backbone = create_RepLKNet31XL(args, pretrained=True)
        self.stem = backbone.stem
        self.stages = backbone.stages
        self.transitions = backbone.transitions
        self.norm = backbone.norm
        self.avgpool = backbone.avgpool
        self.dropout = backbone.dropout
        self.head = backbone.head

    def construct(self, x):
        x = self.stem(x)
        x = self.stages[0](x)

        x = self.transitions[0](x)
        x = self.stages[1](x)

        x = self.transitions[1](x)
        x = self.stages[2](x)

        x = self.transitions[2](x)
        x = self.stages[3](x)

        x = self.norm(x)
        x = self.avgpool(x, [2, 3])
        x = self.dropout(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    import argparse
    from mindspore import context
    import mindspore as ms

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_classes = 1000
    args.drop_rate = 0.
    args.drop_path_rate = 0.2

    img = ms.numpy.ones((2, 3, 224, 224))
    # context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(mode=context.GRAPH_MODE)
    net = MSNet(args)
    print(net)
    out = net(img)
