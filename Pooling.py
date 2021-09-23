
import numpy as np

from math import ceil

from numba import jit, prange

# calculation functions:

def max_pooling(x, params):

    mask_shape = params[0]

    if len(x.shape) != len(mask_shape):
        print("max pooling error: input shape and mask_shape incompatible.")
        exit(1)

    if len(x.shape) not in [1, 2, 3]:
        print("max pooling error: input dimension is bigger than 3.")
        exit(1)

    # else:

    if len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1):
        return max_pooling1D_func(x, mask_shape, down=params[1])
    elif len(x.shape) == 2:
        return max_pooling2D_func(x, mask_shape, right=params[1], down=params[2])
    elif len(x.shape) == 3:
        return max_pooling3D_func(x, mask_shape, right=params[1], down=params[2], back=params[3])


@jit(nopython=True, parallel=True)
def max_pooling1D_func(x, mask_shape, down):

    mask_height = mask_shape
    m = x.shape[0]

    res_shape = (int((m-mask_height) / down + 1, 1), )

    B = np.empty(shape=res_shape)

    for t in prange(0, res_shape[0]):
        B[t, 1] = np.max(x[t*down: t*down+mask_height])

    return B


@jit(nopython=True, parallel=True)
def max_pooling2D_func(x, mask_shape, right, down):

    mask_height, mask_width = mask_shape
    m, n = x.shape

    res_shape = (int((m-mask_height) / down + 1), int((n-mask_width) / right + 1))

    B = np.empty(shape=res_shape)

    for t in prange(0, res_shape[0]):
        for l in prange(0, res_shape[1]):
            B[t, l] = np.max(x[t*down: t*down+mask_height, l*right: l*right+mask_width])

    return B


@jit(nopython=True, parallel=True)
def max_pooling3D_func(x, mask_shape, right, down, back):

    mask_height, mask_width, mask_depth = mask_shape
    m, n, k = x.shape

    res_shape = (int((m-mask_height) / down + 1), int((n-mask_width) / right + 1), int((k-mask_depth) / back + 1))

    B = np.empty(shape=res_shape)

    for t in prange(0, res_shape[0]):
        for l in prange(0, res_shape[1]):
            for e in prange(0, res_shape[2]):
                B[t, l] = np.max(x[t*down: t*down+mask_height, l*right: l*right+mask_width, e*back: e*back+mask_depth])

    return B


# jacobian-vector-product calculation functions:

@jit(nopython=True, parallel=True)
def max_pooling1D_jvp(child_diff, x, y, additional_info):

    mask_height = additional_info[0]
    down = additional_info[1]

    pool_height = y.shape[0]

    res = np.zeros(shape=x.shape)

    for i in prange(0, res.shape[0]):
        row_inset = max(ceil((i - mask_height) / down) + 1, 0)
        row_offset = min(ceil(i / down) + 1, pool_height)

        for t in prange(row_inset, row_offset):

            if (x[i] >= y[t] - 1e-8) and (x[i] <= y[t] + 1e-8):
                res[i] += child_diff[t]

    return res


@jit(nopython=True, parallel=True)
def max_pooling2D_jvp(child_diff, x, y, additional_info):

    mask_height, mask_width = additional_info[0], additional_info[1]
    right, down = additional_info[2], additional_info[3]

    pool_height, pool_width = y.shape

    res = np.zeros(shape=x.shape)

    for i in prange(0, res.shape[0]):
        for j in prange(0, res.shape[1]):

            row_inset = max(ceil((i - mask_height) / down) + 1, 0)
            row_offset = min(ceil(i / down) + 1, pool_height)

            column_inset = max(ceil((j - mask_width) / right) + 1, 0)
            column_offset = min(ceil(j / right) + 1, pool_width)

            for t in prange(row_inset, row_offset):
                for l in prange(column_inset, column_offset):

                    if (x[i, j] >= y[t, l] - 1e-8) and (x[i, j] <= y[t, l] + 1e-8):
                        # y[t, l] is probably the maximum value of the area we are looking at.
                        res[i, j] += child_diff[t, l]

    return res


@jit(nopython=True, parallel=True)
def max_pooling3D_jvp(child_diff, x, y, additional_info):

    mask_height, mask_width, mask_depth = additional_info[0], additional_info[1], additional_info[2]
    right, down, back = additional_info[3], additional_info[4], additional_info[5]

    pool_height, pool_width, pool_depth = y.shape

    res = np.zeros(shape=x.shape)

    for i in prange(0, res.shape[0]):
        for j in prange(0, res.shape[1]):
            for k in prange(0, res.shape[2]):

                row_inset = max(ceil((i - mask_height) / down) + 1, 0)
                row_offset = min(ceil(i / down) + 1, pool_height)

                column_inset = max(ceil((j - mask_width) / right) + 1, 0)
                column_offset = min(ceil(j / right) + 1, pool_width)

                depth_inset = max(ceil((k - mask_depth) / back) + 1, 0)
                depth_offset = min(ceil(k / back) + 1, pool_depth)

                for t in prange(row_inset, row_offset):
                    for l in prange(column_inset, column_offset):
                        for e in prange(depth_inset, depth_offset):

                            if (x[i, j, k] >= y[t, l, e] - 1e-8) and (x[i, j, k] <= y[t, l, e] + 1e-8):
                                res[i, j, k] += child_diff[t, l, e]

    return res
