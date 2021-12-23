from collections import namedtuple

import torch

Genotype = namedtuple('Genotype', 'reduce reduce_concat up up_concat')

PRIMITIVES_DOWN = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

PRIMITIVES_UP = [
    'none_trans',
    'skip_connect_trans',
    'sep_conv_trans_3x3',
    'sep_conv_trans_5x5',
    'sep_conv_trans_7x7',
    'dil_conv_trans_3x3',
    'dil_conv_trans_5x5'
]

NAS_StegaNet = Genotype(
    reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 1), ('sep_conv_7x7', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 3)],
    reduce_concat=range(2, 6),
    up=[('dil_conv_trans_3x3', 1), ('dil_conv_trans_3x3', 0), ('skip_connect_trans', 1), ('sep_conv_trans_3x3', 2),
        ('dil_conv_trans_3x3', 1), ('sep_conv_trans_3x3', 2), ('dil_conv_trans_3x3', 1), ('dil_conv_trans_5x5', 2)],
    up_concat=range(2, 6))

