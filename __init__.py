
from AutoDiff.BasicGradients import defjvp
from AutoDiff.Operators import Op
import AutoDiff.convolution
from AutoDiff.Pooling import max_pooling1D_jvp, max_pooling2D_jvp, max_pooling3D_jvp

import numpy as np


defjvp(Op.add,
       lambda child_diff, x1, x2, y, additional_info=None: child_diff,
       lambda child_diff, x1, x2, y, additional_info=None: child_diff)

defjvp(Op.subtract,
       lambda child_diff, x1, x2, y, additional_info=None: child_diff,
       lambda child_diff, x1, x2, y, additional_info=None: -1*child_diff)

defjvp(Op.true_division,
       lambda child_diff, x1, x2, y, additional_info=None: child_diff / x2,
       lambda child_diff, x1, x2, y, additional_info=None: -child_diff * (y/(x2+1e-8)))

defjvp(Op.elem_wise_mult,
       lambda child_diff, x1, x2, y, additional_info=None: child_diff * x2,
       lambda child_diff, x1, x2, y, additional_info=None: child_diff * x1)

defjvp(Op.power,
       lambda child_diff, x1, x2, y, additional_info: child_diff * x2 * y / (x1+1e-8),
       lambda child_diff, x1, x2, y, additional_info: child_diff * y * np.log(x1))

defjvp(Op.matmul,
       lambda child_diff, x1, x2, y, additional_info=None: np.matmul(child_diff, x2.T),
       lambda child_diff, x1, x2, y, additional_info=None: np.matmul(x1.T, child_diff))

defjvp(Op.neg,
       lambda child_diff, x, y, additional_info=None: -1*child_diff)

defjvp(Op.ge,
       lambda child_diff, x1, x2, y, additional_info: 0,
       lambda child_diff, x1, x2, y, additional_info: 0)

defjvp(Op.conv_2D,
       convolution.conv2D_jvp_first_arg,
       convolution.conv2D_jvp_second_arg)

defjvp(Op.conv_3D,
       convolution.conv3D_jvp_first_arg,
       convolution.conv3D_jvp_second_arg)

def concat_jvp_arg1(child_diff, x1, x2, y, additional_info=None):

       return child_diff[:x1.shape[0], :]

def concat_jvp_arg2(child_diff, x1, x2, y, additional_info=None):

       return child_diff[x1.shape[0] : , :]

defjvp(Op.concat_vectors,
       lambda child_diff, x1, x2, y, additional_info: child_diff[:x1.shape[0], :],
       lambda child_diff, x1, x2, y, additional_info: child_diff[x1.shape[0]: , :])

defjvp(Op.concat_vectors,
       concat_jvp_arg1,
       concat_jvp_arg2)

defjvp(Op.exp,
       lambda child_diff, x, y, additional_info=None: child_diff * y)

defjvp(Op.log,
       lambda child_diff, x, y, additional_info=None: child_diff / (x+1e-8))

defjvp(Op.sum,
       lambda child_diff, x, y, additional_info: child_diff * np.ones(x.shape))

defjvp(Op.reshape,
       lambda child_diff, x, y, additional_info: np.reshape(child_diff, newshape=additional_info[0]))

defjvp(Op.dup,
       lambda child_diff, x, y, additional_info: np.sum(child_diff))

defjvp(Op.abs,
       lambda child_diff, x, y, additional_info: child_diff * (2*(x > 0) - 1))

defjvp(Op.sin,
       lambda child_diff, x, y, additional_info: child_diff * np.cos(x))

defjvp(Op.maxpool_1D,
       Pooling.max_pooling1D_jvp)

defjvp(Op.maxpool_2D,
       Pooling.max_pooling2D_jvp)

defjvp(Op.maxpool_3D,
       Pooling.max_pooling3D_jvp)

defjvp(Op.relu,
       lambda child_diff, x, y, additional_info: child_diff * (x > 0))

defjvp(Op.softmax,
       lambda child_diff, x, y, additional_info: -np.sum(child_diff * y) * y + child_diff*y)

defjvp(Op.sigmoid,
       lambda child_diff, x, y, additional_info: child_diff * y * (1-y))

defjvp(Op.tanh,
       lambda child_diff, x, y, additional_info: child_diff * (1 - (y ** 2)))