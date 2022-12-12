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

# s91_8 = Genotype(normal=[('max_pool_3x3', 0), ('bin_dil_conv_5x5', 1), ('max_pool_3x3', 1), ('bin_conv_3x3', 0), ('bin_conv_3x3', 1), ('bin_conv_3x3', 0), ('skip_connect', 0), ('bin_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('bin_conv_5x5', 0), ('bin_conv_3x3', 1), ('bin_dil_conv_3x3', 2), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_3x3', 3), ('bin_conv_3x3', 3), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

# s91_9 = Genotype(normal=[('bin_dil_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('bin_dil_conv_5x5', 0), ('max_pool_3x3', 0), ('bin_conv_3x3', 1), ('bin_conv_3x3', 1), ('bin_dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('bin_conv_3x3', 0), ('bin_dil_conv_5x5', 1), ('skip_connect', 0), ('bin_dil_conv_5x5', 3), ('bin_conv_3x3', 1), ('bin_conv_3x3', 1), ('bin_conv_3x3', 0)], reduce_concat=range(2, 6))

# s91_10 = Genotype(normal=[('bin_dil_conv_5x5', 1), ('bin_dil_conv_5x5', 0), ('bin_dil_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('bin_dil_conv_3x3', 0), ('skip_connect', 1), ('bin_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('bin_conv_5x5', 0), ('bin_dil_conv_5x5', 2), ('bin_conv_5x5', 1), ('bin_conv_3x3', 0), ('bin_dil_conv_5x5', 3), ('bin_conv_3x3', 1), ('bin_dil_conv_5x5', 4)], reduce_concat=range(2, 6))

# s91_16 = Genotype(normal=[('max_pool_3x3', 0), ('bin_conv_5x5', 1), ('bin_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('bin_conv_5x5', 1), ('bin_dil_conv_5x5', 0), ('bin_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('bin_dil_conv_5x5', 0), ('bin_conv_5x5', 1), ('bin_conv_5x5', 2), ('bin_conv_3x3', 1), ('bin_dil_conv_3x3', 2), ('bin_dil_conv_3x3', 3), ('bin_dil_conv_3x3', 2), ('bin_dil_conv_3x3', 3)], reduce_concat=range(2, 6))

s91_16b = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('bin_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('bin_conv_5x5', 1), ('bin_dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('bin_conv_5x5', 0), ('bin_dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('bin_conv_3x3', 1), ('bin_dil_conv_3x3', 3), ('bin_dil_conv_5x5', 4), ('bin_dil_conv_3x3', 1)], reduce_concat=range(2, 6)) #94.04 2.944490MB

s91_16b2 = Genotype(normal=[('bin_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('bin_conv_3x3', 1), ('bin_conv_5x5', 0), ('max_pool_3x3', 0), ('bin_dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 0), ('bin_conv_5x5', 1), ('bin_conv_3x3', 1), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_5x5', 2), ('bin_conv_3x3', 1), ('max_pool_3x3', 0), ('bin_conv_3x3', 1)], reduce_concat=range(2, 6)) #94.09 3.077378MB

s91_16b3 = Genotype(normal=[('max_pool_3x3', 0), ('bin_conv_3x3', 1), ('max_pool_3x3', 1), ('bin_dil_conv_3x3', 0), ('bin_dil_conv_3x3', 1), ('avg_pool_3x3', 2), ('bin_conv_3x3', 1), ('bin_dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 0), ('max_pool_3x3', 1), ('bin_conv_3x3', 0), ('max_pool_3x3', 2), ('bin_dil_conv_5x5', 3), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_3x3', 4), ('bin_conv_5x5', 1)], reduce_concat=range(2, 6)) #94.00 3.168266MB

# s91_16one = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('bin_conv_5x5', 0), ('bin_conv_3x3', 0), ('bin_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('bin_dil_conv_5x5', 0), ('bin_conv_5x5', 1), ('bin_dil_conv_5x5', 2), ('bin_conv_5x5', 1), ('bin_dil_conv_3x3', 3), ('bin_dil_conv_3x3', 2), ('bin_dil_conv_3x3', 4), ('bin_conv_3x3', 1)], reduce_concat=range(2, 6))

# s91_16two = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('bin_dil_conv_5x5', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('bin_conv_3x3', 0), ('bin_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('bin_dil_conv_5x5', 1), ('bin_conv_3x3', 0), ('bin_dil_conv_5x5', 2), ('bin_conv_3x3', 0), ('bin_dil_conv_3x3', 1), ('bin_dil_conv_5x5', 3), ('bin_dil_conv_5x5', 4), ('bin_conv_3x3', 1)], reduce_concat=range(2, 6))

# s91_16three = Genotype(normal=[('bin_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('bin_conv_3x3', 1), ('bin_dil_conv_3x3', 0), ('bin_dil_conv_3x3', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('bin_conv_3x3', 1), ('bin_conv_3x3', 0), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_3x3', 3), ('bin_conv_3x3', 1), ('bin_conv_5x5', 4)], reduce_concat=range(2, 6))

# s91_16four = Genotype(normal=[('bin_dil_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('bin_conv_3x3', 1), ('bin_conv_3x3', 1), ('bin_dil_conv_3x3', 0), ('bin_dil_conv_3x3', 0), ('bin_dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('bin_conv_5x5', 0), ('bin_dil_conv_3x3', 2), ('bin_conv_5x5', 1), ('bin_dil_conv_5x5', 3), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_3x3', 3), ('bin_conv_3x3', 1)], reduce_concat=range(2, 6))


# s1a = Genotype(normal=[('bin_dil_conv_5x5', 1), ('max_pool_3x3', 0), ('bin_dil_conv_5x5', 1), ('bin_dil_conv_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('bin_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 0), ('bin_dil_conv_5x5', 1), ('bin_conv_3x3', 0), ('max_pool_3x3', 2), ('bin_dil_conv_3x3', 3), ('bin_dil_conv_3x3', 2), ('bin_dil_conv_3x3', 3), ('bin_dil_conv_3x3', 4)], reduce_concat=range(2, 6))

# s1b = Genotype(normal=[('bin_dil_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('bin_dil_conv_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('bin_dil_conv_3x3', 1), ('bin_dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('skip_connect', 0), ('bin_conv_3x3', 0), ('bin_dil_conv_5x5', 2), ('max_pool_3x3', 2), ('bin_dil_conv_3x3', 3), ('bin_dil_conv_3x3', 1), ('bin_conv_3x3', 0)], reduce_concat=range(2, 6))

# s1c = Genotype(normal=[('max_pool_3x3', 0), ('bin_conv_3x3', 1), ('max_pool_3x3', 0), ('bin_conv_3x3', 1), ('bin_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('bin_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('skip_connect', 0), ('bin_dil_conv_5x5', 2), ('bin_conv_3x3', 1), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_5x5', 3), ('bin_dil_conv_5x5', 2), ('bin_conv_3x3', 0)], reduce_concat=range(2, 6))

# s88 = Genotype(normal=[('skip_connect', 0), ('bin_dil_conv_5x5', 1), ('bin_conv_3x3', 1), ('bin_dil_conv_3x3', 2), ('bin_dil_conv_3x3', 1), ('bin_dil_conv_3x3', 0), ('bin_dil_conv_3x3', 0), ('bin_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('bin_conv_5x5', 0), ('bin_dil_conv_3x3', 1), ('max_pool_3x3', 0), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_3x3', 4), ('bin_dil_conv_3x3', 2)], reduce_concat=range(2, 6))

# s91_10 = Genotype(normal=[('bin_dil_conv_5x5', 1), ('bin_dil_conv_5x5', 0), ('bin_dil_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('bin_dil_conv_3x3', 0), ('skip_connect', 1), ('bin_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('bin_conv_5x5', 0), ('bin_dil_conv_5x5', 2), ('bin_conv_5x5', 1), ('bin_conv_3x3', 0), ('bin_dil_conv_5x5', 3), ('bin_conv_3x3', 1), ('bin_dil_conv_5x5', 4)], reduce_concat=range(2, 6))

# a1 = Genotype(normal=[('bin_dil_conv_5x5', 1), ('max_pool_3x3', 0), ('bin_dil_conv_5x5', 1), ('bin_conv_3x3', 0), ('bin_dil_conv_3x3', 0), ('bin_dil_conv_3x3', 1), ('bin_dil_conv_3x3', 0), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('bin_conv_3x3', 0), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_3x3', 1), ('bin_dil_conv_5x5', 3), ('max_pool_3x3', 2), ('bin_dil_conv_3x3', 4), ('bin_conv_3x3', 3)], reduce_concat=range(2, 6)) # w path w/o edge

# a2 = Genotype(normal=[('bin_dil_conv_5x5', 0), ('max_pool_3x3', 1), ('bin_dil_conv_5x5', 1), ('bin_conv_3x3', 0), ('bin_dil_conv_3x3', 0), ('bin_dil_conv_3x3', 1), ('bin_conv_3x3', 1), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('max_pool_3x3', 0), ('bin_conv_3x3', 1), ('bin_dil_conv_5x5', 2), ('bin_dil_conv_5x5', 3), ('bin_dil_conv_3x3', 2), ('bin_dil_conv_3x3', 3), ('bin_dil_conv_3x3', 4)], reduce_concat=range(2, 6)) # w path w/o edge

# a3 = Genotype(normal=[('bin_dil_conv_5x5', 1), ('max_pool_3x3', 0), ('bin_dil_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('bin_dil_conv_5x5', 0), ('bin_conv_3x3', 1), ('bin_dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('bin_conv_5x5', 1), ('bin_dil_conv_5x5', 0), ('bin_dil_conv_5x5', 2), ('bin_conv_5x5', 1), ('bin_dil_conv_5x5', 3), ('skip_connect', 1), ('bin_dil_conv_5x5', 4), ('bin_dil_conv_5x5', 1)], reduce_concat=range(2, 6)) # w/o path w edge

# a4 = Genotype(normal=[('bin_dil_conv_5x5', 1), ('bin_dil_conv_5x5', 0), ('bin_dil_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('bin_conv_3x3', 0), ('bin_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('bin_dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('bin_dil_conv_5x5', 2), ('bin_conv_3x3', 1), ('bin_dil_conv_5x5', 3), ('bin_conv_5x5', 1), ('bin_dil_conv_5x5', 4), ('bin_conv_3x3', 0)], reduce_concat=range(2, 6)) # w/o path w edge
