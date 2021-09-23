from math import ceil
import numpy as np

from numba import jit, prange


# convolution operations:


@jit(nopython=True, parallel=True)
def conv2D_func(A, W, right, down):

    """gets 2D numpy.ndarrays and returns the result of a convolution on them."""

    # if len(A.shape) != 2 or len(W.shape) != 2:
    #     print("Invalid Parameters for a 2D convolution.")
    #     exit(1)
    #
    # for i in range(0, 2):
    #     if W.shape[i] > A.shape[i]:
    #         print("conv2D_error: Mask parameter has a dimension which's bigger than A's same dimension.")
    #         exit(1)

    # else:

    # check if any padding is needed, and add it.

    m, n = A.shape
    mask_height, mask_width = W.shape
    r, d = right, down

    # after padding:

    conv_height = int((m-mask_height) / d + 1)
    conv_width = int((n-mask_width) / r + 1)

    B = np.zeros(shape=(conv_height, conv_width))

    for i in prange(conv_height):
        for j in prange(conv_width):

            for a in prange(0, mask_height):
                for b in prange(0, mask_width):
                    B[i, j] += A[i*d+a, j*r+b] * W[a, b]

    return B


@jit(nopython=True, parallel=True)
def conv3D_func(A, W, right, down, back):

    """gets 2D numpy.ndarrays and returns the result of a convolution on them."""

    # if len(A.shape) != 3 or len(W.shape) != 3:
    #     print("Invalid Parameters for a 3D convolution.")
    #     exit(1)
    #
    # for i in range(0, 3):
    #     if W.shape[i] > A.shape[i]:
    #         print("conv2D_error: Mask parameter has a dimension which's bigger than A's same dimension.")
    #         exit(1)

    # else:

    m, n, l = A.shape
    mask_height, mask_width, mask_depth = W.shape

    # check if any padding is needed, and add it.

    # after padding:

    conv_height = int((m-mask_height) / down + 1)
    conv_width = int((n-mask_width) / right + 1)
    conv_depth = int((l-mask_depth) / back + 1)

    B = np.zeros((conv_height, conv_width, conv_depth))

    for i in prange(conv_height):
        for j in prange(conv_width):
            for k in prange(conv_depth):

                for a in prange(0, mask_height):
                    for b in prange(0, mask_width):
                        for c in prange(0, mask_depth):
                            B[i, j, k] += A[i*down+a, j*right+b, k*back+c] * W[a, b, c]

    return B


# convolution jacobian vector product operations:

@jit(nopython=True, parallel=True)
def conv2D_jvp_first_arg(child_diff, x1, x2, y, additional_info):

    res = np.zeros(shape=x1.shape)

    mask_height, mask_width = x2.shape
    conv_height, conv_width = y.shape

    right, down = additional_info[0], additional_info[1]

    for i in prange(0, x1.shape[0]):
        for j in prange(0, x1.shape[1]):

            row_inset = max(ceil((i-mask_height) / down) + 1, 0)
            row_offset = min(ceil(i / down) + 1, conv_height)

            column_inset = max(ceil((j-mask_width) / right) + 1, 0)
            column_offset = min(ceil(j / right) + 1, conv_width)

            for t in prange(row_inset, row_offset):
                for l in prange(column_inset, column_offset):
                    res[i, j] += child_diff[t, l] * x2[i - t*down, j - l*right]

    return res


def conv2D_jvp_second_arg(child_diff, x1, x2, y, additional_info):

    m, n = x1.shape
    mask_height, mask_width = x2.shape

    right, down = additional_info[0], additional_info[1]

    res = np.empty(shape=x2.shape)

    for a in prange(x2.shape[0]):
        for b in prange(x2.shape[1]):

            cut = x1[a:m-mask_height+a+1:down, b:n-mask_width+b+1:right]

            res[a, b] = np.sum(child_diff * cut)

    return res


@jit(nopython=True, parallel=True)
def conv2D_jvp_2(child_diff, x1, x2, y, additional_info):

    res = np.zeros(x2.shape)

    conv_height, conv_width = y.shape
    right, down = additional_info[0], additional_info[1]

    for a in prange(0, x2.shape):
        for b in prange(0, x2.shape):

            for t in prange(0, conv_height):
                for l in prange(0, conv_width):
                    res[a, b] += child_diff[t, l] * x1[t*down + a, l*right + b]

    return res


@jit(nopython=True, parallel=True)
def conv3D_jvp_first_arg(child_diff, x1, x2, y, additional_info):

    res = np.zeros(shape=x1.shape)

    conv_height, conv_width, conv_depth = x2.shape
    right, down, back = additional_info[0], additional_info[1], additional_info[2]

    for i in prange(0, x1.shape[0]):
        for j in prange(0, x1.shape[1]):
            for k in prange(0, x1.shape[2]):

                row_inset = ceil((i-conv_height) / down) + 1
                row_offset = ceil(i / down) + 1

                column_inset = ceil((j-conv_width) / right) + 1
                column_offset = ceil(j / right) + 1

                depth_inset = ceil((k-conv_depth) / back) + 1
                depth_offset = ceil(k / back) + 1

                for t in range(row_inset, row_offset):
                    for l in range(column_inset, column_offset):
                        for e in range(depth_inset, depth_offset):

                            res[i, j, k] += child_diff[t, l, e] * x2[i-t*down, j-l*right, k-e*back]

    return res


@jit(nopython=True, parallel=True)
def conv3D_jvp_second_arg(child_diff, x1, x2, y, additional_info):

    m, n, k = x1.shape
    mask_height, mask_width, mask_depth = x2.shape

    right, down, back = additional_info[0], additional_info[1], additional_info[2]

    res = np.empty(x2.shape)

    for a in prange(x2.shape[0]):
        for b in prange(x2.shape[1]):
            for c in prange(x2.shape[2]):

                cut = x1[a:m-mask_height+a+1:down, b:n-mask_width+b+1:right, c: k-mask_depth+c+1:back]

                res[a, b, c] = np.sum(child_diff * cut)

    return res


@jit(nopython=True, parallel=True)
def conv3D_jvp_2(child_diff, x1, x2, y, additional_info):

    res = np.zeros(x2.shape)

    conv_height, conv_width, conv_depth = y.shape
    right, down, back = additional_info[0], additional_info[1], additional_info[2]

    for a in prange(0, x2.shape):
        for b in prange(0, x2.shape):
            for c in prange(0, x2.shape):

                for t in range(0, conv_height):
                    for l in range(0, conv_width):
                        for e in range(0, conv_depth):
                            res[a, b, c] += child_diff[t, l, e] * x1[t*down + a, l*right + b, e*back + c]

    return res
