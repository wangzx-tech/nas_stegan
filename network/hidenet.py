# encoding: utf-8

import functools

import torch
import torch.nn as nn


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
from network.operations import ReLUConvBN, OPS, FactorizedReduce, ReLUConvTransBN


class UnetG(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False, out_func=nn.Tanh):
        super(UnetG, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, output_function=out_func)

        self.model = unet_block
        # self.tanh = out_func == nn.Tanh
        if out_func == nn.Tanh:
            self.factor = 10 / 255
        else:
            self.factor = 1.0

    def forward(self, input):
        return self.factor * self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=None, use_dropout=False, output_function=nn.Tanh):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer is None:
            use_bias = True
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        # modify classic Conv to dep conv
        # downconv = nn.Sequential(
        #     nn.Conv2d(input_nc, input_nc, kernel_size=4, stride=2, padding=1, groups=input_nc, bias=use_bias),
        #     nn.Conv2d(input_nc, inner_nc, kernel_size=1, padding=0, bias=use_bias),
        # )

        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        if norm_layer is not None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # upconv = nn.Sequential(
            #     nn.ConvTranspose2d(inner_nc * 2, inner_nc * 2, kernel_size=4, stride=2, padding=1, groups=inner_nc * 2),
            #     nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=1, padding=0),
            # )

            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            # upconv = nn.Sequential(
            #     nn.ConvTranspose2d(inner_nc, inner_nc, kernel_size=4, stride=2, padding=1, groups=inner_nc, bias=use_bias),
            #     nn.Conv2d(inner_nc, outer_nc, kernel_size=1, padding=0, bias=use_bias),
            # )

            down = [downrelu, downconv]
            if norm_layer is None:
                up = [uprelu, upconv]
            else:
                up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            # upconv = nn.Sequential(
            #     nn.ConvTranspose2d(inner_nc * 2, inner_nc * 2, kernel_size=4, stride=2, padding=1,
            #                        groups=inner_nc * 2, bias=use_bias),
            #     nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=1, padding=0, bias=use_bias),
            # )

            if norm_layer is None:
                down = [downrelu, downconv]
                up = [uprelu, upconv]
            else:
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class NASG(nn.Module):

    def __init__(self, C, layers, genotype, in_channel, out_channel):
        super(NASG, self).__init__()
        self._layers = layers

        self.stem0 = nn.Sequential(
            nn.Conv2d(in_channel, C, 7, padding=3, bias=False),
            nn.BatchNorm2d(C)
        )

        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        # Flag control cell class
        downsamp_prev = 0
        for i in range(layers):
            if i < layers // 2:
                # Downsampling cell: imagesize/2
                # -1:Downsampling cell, 1:Upsampling cell
                cell_class = -1
            else:
                # Upsampling cell: imagesize*2
                cell_class = 1

            # update cell
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, cell_class, downsamp_prev)
            downsamp_prev = cell_class
            # update cells list
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_prev, out_channel, 3, 1, 1),
        )

        self.out_func = nn.Tanh()

    def forward(self, input):

        s0 = s1 = self.stem0(input)

        # Save downsampling cells' states
        down_states = [s1]

        for i, cell in enumerate(self.cells):
            if cell.cell_class == -1:
                s0, s1 = s1, cell(s0, s1)
                down_states.append(s1)
            elif cell.cell_class == 1:
                s0 = down_states[-(i + 2 - len(down_states))]
                s1 = cell(s0, s1)
        output = self.stem1(s1)
        stega_se = (10 / 255) * self.out_func(output)

        return stega_se


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, cell_class, downsamp_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.cell_class = cell_class

        if downsamp_prev == -1 and cell_class == -1:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        elif downsamp_prev == 0 or cell_class == 1:
            self.preprocess0 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if cell_class == -1:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        elif cell_class == 1:
            op_names, indices = zip(*genotype.up)
            concat = genotype.up_concat
        self._compile(C, op_names, indices, concat, cell_class)

    # Combine cells' node to form a network
    def _compile(self, C, op_names, indices, concat, cell_class):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            if abs(cell_class) == 1 and index < 2:
                stride = 2
            else:
                stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)
