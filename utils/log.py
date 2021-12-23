import os

import numpy as np
from thop import profile


def print_net(net, logpath, size):
    # count parameters
    macs, params = profile(net, inputs=(size, ))
    flops = 2 * macs

    print(str(net))
    save_log(str(net), logpath)

    print(f'Total number of parameters: {params/1000**2} M and flops: {flops/1000**3} G')
    save_log(f'Total number of parameters: {params/1000**2} M and flops: {flops/1000**3} G', logpath)


def save_log(info, path):
    if not os.path.exists(path):
        fp = open(path, "w")
        fp.writelines(info + "\n")
    else:
        with open(path, 'a+') as f:
            f.writelines(info + '\n')
