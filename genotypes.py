from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'skip_connect',
    'max_pool_3x3',
    'avg_pool_3x3',
    'bin_conv_3x3',
    'bin_conv_5x5',
    'bin_dil_conv_3x3',
    'bin_dil_conv_5x5'
]

EBNAS = Genotype(normal=[('bin_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('bin_conv_3x3', 1), ('bin_conv_5x5', 0), ('max_pool_3x3', 0), ('bin_dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 0), ('bin_conv_5x5', 1), ('bin_conv_3x3', 1), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_5x5', 2), ('bin_conv_3x3', 1), ('max_pool_3x3', 0), ('bin_conv_3x3', 1)], reduce_concat=range(2, 6))


