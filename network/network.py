import torch
import torch.nn.functional as F
from network.genotypes import PRIMITIVES_DOWN, PRIMITIVES_UP
from network.genotypes import Genotype
from torch import nn

from network.operations import ReLUConvBN, OPS, FactorizedReduce, ReLUConvTransBN


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module):

    def __init__(self, C, stride, cell_class):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.k = 4
        self.mp = nn.MaxPool2d(2, 2)
        self.ump = nn.UpsamplingNearest2d(scale_factor=2)
        # Operations that form each cell
        if cell_class == -1:
            primitives = PRIMITIVES_DOWN
        elif cell_class == 1:
            primitives = PRIMITIVES_UP
        for primitive in primitives:
            op = OPS[primitive](C // self.k, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // self.k, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        # channel proportion k = 4
        dim_2 = x.shape[1]
        xtemp = x[:, :dim_2 // self.k, :, :]
        xtemp2 = x[:, dim_2 // self.k:, :, :]

        # Only calculate the top 1/4 features
        temp1 = sum(w.to(xtemp.device) * op(xtemp) for w, op in zip(weights, self._ops))

        # When cells change dimensions, down-sampling and up-sampling are required to ensure the same dimensions
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        elif temp1.shape[2] < x.shape[2]:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        else:
            ans = torch.cat([temp1, self.ump(xtemp2)], dim=1)

        # Disrupt the output channel
        ans = channel_shuffle(ans, self.k)
        return ans


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, cell_class, downsamp_prev):
        super(Cell, self).__init__()
        self.cell_class = cell_class
        # The first two layers are fixed to 1*1 convolution to ensure dimensionality
        if downsamp_prev == -1 and cell_class == -1:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        elif downsamp_prev == 0 or cell_class == 1:
            self.preprocess0 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps  # 4
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        # Build a mixed operation of four nodes in sequence
        for i in range(self._steps):
            for j in range(2 + i):
                if abs(cell_class) == 1 and j < 2:
                    stride = 2
                else:
                    stride = 1
                # MixedOp builds a mixed operation
                op = MixedOp(C, stride, cell_class)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        # Calculate the output of the four nodes in sequence and add them to the states
        for i in range(self._steps):
            s = sum(weights2[offset + j].to(self._ops[offset + j](h, weights[offset + j]).device)
                    * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        # concat range[2, 6]
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C  # initial channels number
        self._layers = layers  # layers number
        self._criterion = criterion  # loss function
        self._steps = steps  # 4 steps for each cell
        self._multiplier = multiplier

        C_curr = stem_multiplier * C

        self.stem_down0 = nn.Sequential(
            nn.Conv2d(3, C_curr, 7, padding=3, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        # update channels' number
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # set a new modulelist object
        self.cells = nn.ModuleList()
        downsamp_prev = 0
        for i in range(layers):
            if i < layers // 2:
                # Downsampling cell: imagesize / 2
                # -1:Downsampling cell, 1:Upsampling cell
                cell_class = -1
            else:
                # Upsampling cell: imagesize * 2
                cell_class = 1

            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, cell_class, downsamp_prev)  # update cell
            downsamp_prev = cell_class
            # update cells list
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.out_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self._C * self._multiplier, 3, 3, 1, 1)
        )
        self.out_func = nn.Tanh()
        # initial architecture weights
        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.item().copy_(y.item())
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem_down0(input)

        # Save down-sampling cells' states
        down_states = [s1]

        for i, cell in enumerate(self.cells):
            if cell.cell_class == -1:
                weights = F.softmax(self.alphas_reduce, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
                for j in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
                # update s0 & s1, add s1 to states list
                s0, s1 = s1, cell(s0, s1, weights, weights2)
                down_states.append(s1)

            elif cell.cell_class == 1:
                weights = F.softmax(self.alphas_up, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_up[0:2], dim=-1)
                for j in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_up[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
                # Use states list to update s0
                s0 = down_states[-(i + 2 - len(down_states))]
                s1 = cell(s0, s1, weights, weights2)

        output = self.out_layer(s1)
        stega_se = (10 / 255) * self.out_func(output)
        return stega_se

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops_reduce = len(PRIMITIVES_DOWN)
        num_ops_up = len(PRIMITIVES_UP)

        self.alphas_reduce = 1e-3 * torch.randn(k, num_ops_reduce).cuda()
        self.alphas_reduce.requires_grad = True
        self.alphas_up = 1e-3 * torch.randn(k, num_ops_up).cuda()
        self.alphas_up.requires_grad = True
        self.betas_reduce = 1e-3 * torch.randn(k).cuda()
        self.betas_reduce.requires_grad = True
        self.betas_up = 1e-3 * torch.randn(k).cuda()
        self.betas_up.requires_grad = True
        self._arch_parameters = [
            self.alphas_reduce,
            self.alphas_up,
            self.betas_reduce,
            self.betas_up,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights, weights2, PRIMITIVES):
            # make a empty gene
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                for x in range(len(PRIMITIVES)):
                    if 'none' in PRIMITIVES[x]:
                        index = x
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k]
                                                                for k in range(len(W[x]))
                                                                if k != index))[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != index:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        n = 3
        start = 2
        weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        weightsu2 = F.softmax(self.betas_up[0:2], dim=-1)
        for i in range(self._steps - 1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            tu2 = F.softmax(self.betas_up[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsu2 = torch.cat([weightsu2, tu2], dim=0)
        # gain gene
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).detach().cpu().numpy(),
                             weightsr2.detach().cpu().numpy(), PRIMITIVES=PRIMITIVES_DOWN)
        gene_up = _parse(F.softmax(self.alphas_up, dim=-1).detach().cpu().numpy(),
                         weightsu2.detach().cpu().numpy(), PRIMITIVES=PRIMITIVES_UP)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)

        genotype = Genotype(
            reduce=gene_reduce, reduce_concat=concat,
            up=gene_up, up_concat=concat
        )
        return genotype
