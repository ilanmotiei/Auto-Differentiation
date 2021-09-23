
from enum import Enum


class Op(Enum):
    add = 1
    subtract = 2
    elem_wise_mult = 3
    matmul = 4
    true_division = 5
    neg = 6
    ge = 7
    exp = 8
    log = 9
    power = 10
    sum = 11
    conv_2D = 12
    conv_3D = 13
    maxpool_1D = 14
    maxpool_2D = 15
    maxpool_3D = 16
    reshape = 17
    dup = 18
    relu=19
    sigmoid=20
    tanh=21
    softmax=22
    concat_vectors = 23
    abs = 24
    sin = 25

