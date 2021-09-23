
import numpy as np
from AutoDiff.convolution import conv2D_func, conv3D_func
import AutoDiff.BasicGradients as bg
from AutoDiff.Operators import Op
from AutoDiff.Pooling import max_pooling1D_func, max_pooling2D_func, max_pooling3D_func
import Activations.Activations as activate

class Node(object):

    # computation graph construction functions (building nodes and connections):

    def __init__(self, np_array, is_parameter=False, is_constant=False):

        self.value = np_array

        self.parents = []

        self.operation = None  # The operation for the node parents which gave the current node.

        self.is_constant = is_constant

        self.diff = None  # Stores the differential for memoization.

        self.optional_info = None

        self.is_param = is_parameter  # Mentions whether the node is a changeable argument of the computation graph.

        self.childs = []

        # used for a dfs run on the graph
        self.color = 0

        # used for gradient computation:
        self.is_relevant = False

        # used for debugging
        self.name = None

    def __connect__(self, *parents_nodes):

        for parent in parents_nodes:
            self.parents.append(parent)
            parent.childs.append(self)

        self.is_constant = np.all([parent.is_constant for parent in parents_nodes])
        # self is a constant iff all of its parents are constants

    @staticmethod
    def __boradcast_nodes__(array1, array2):
        broadcasting_shape = np.broadcast_shapes(array1, array2)

        node1 = Node(np_array=np.broadcast_to(array1, shape=broadcasting_shape))
        node2 = Node(np_array=np.broadcast_to(array2, shape=broadcasting_shape))

        return node1, node2

    @staticmethod
    def const_val(value, shape):
        node_value = value * np.ones(shape)

        node = Node(node_value)

        node.is_constant = True

        return node

    # functions for getting information on the computation graph:

    def __topological_sort__(self):

        """
        sorts the nodes at the graph that "self" node depends on, including "self", in a topological order.
        if that can't be done, raises an error.
        """

        finishing_time_list = []

        if self.__dfs_visit__(finishing_list=finishing_time_list) == -1:
            print("Computation failed due to a cyclic dependency in the given function.")
            exit(1)

        # else:
        finishing_time_list.reverse()

        return finishing_time_list

    def __dfs_visit__(self, finishing_list):

        self.color = 1
        self.is_relevant = True  # now we will look at this value when computing the gradient.

        for parent in self.parents:
            if parent.color == 1:
                # A cycle was detected.
                return -1
            if parent.color == 2:
                # parent has already executed the dfs_visit function.
                continue
            if parent.__dfs_visit__(finishing_list) == -1:
                return -1

        self.color = 2

        finishing_list.append(self)  # append self to the end of the list

    def __is_first_arg_of__(self, other):

        if other.parents[0] is self:
            return True

        # else
        return False

    def __is_second_arg_of__(self, other):

        return (len(other.parents) == 2) and (other.parents[1] is self)

    # functions that changes states of nodes at the graph:

    def def_as_param(self):
        self.is_param = True

    def reset_node(self):
        """
        resets all of the information the node holds which is related to the computation graph it is in.
        the method does not reset the 'value' field of the node, and its state (const, parameter, etc.)
        """

        self.parents = []
        self.childs = []
        self.is_relevant = False
        self.optional_info = []
        self.operation = None
        self.diff = None
        self.is_relevant = False
        self.color = 0

    # more functions that construct the computation graph:

    @staticmethod
    def operation_template(a, func, operation, b=None, optional_info=None, optional_params=None):

        if b is None:
            if optional_params is None:
                result_value = func(a.value)
            else:
                result_value = func(a.value, optional_params)
        else:
            if optional_params is None:
                result_value = func(a.value, b.value)
            else:
                result_value = func(a.value, b.value, optional_params)

        result = Node(result_value)

        if b is None:
            result.__connect__(a)
        else:
            result.__connect__(a, b)

        result.operation = operation

        result.optional_info = optional_info

        return result

    # binary operations:

    def __add__(self, other):

        return Node.operation_template(a=self, b=other, func=np.ndarray.__add__, operation=Op.add)

    def __sub__(self, other):

        return Node.operation_template(a=self, b=other, func=np.ndarray.__sub__, operation=Op.subtract)

    def __mul__(self, other):

        return Node.operation_template(a=self, b=other, func=np.ndarray.__mul__, operation=Op.elem_wise_mult)

    def __truediv__(self, other):

        return Node.operation_template(a=self, b=other, func=lambda x, y: x/(y+1e-8), operation=Op.true_division)

    def __pow__(self, power):

        return Node.operation_template(a=self, b=power, func=np.ndarray.__pow__, operation=Op.power)

    def __ge__(self, other):

        return Node.operation_template(a=self, b=other, func=np.ndarray.__ge__, operation=Op.ge)

    def matmul(self, other):

        return Node.operation_template(a=self, b=other,
                                       func=np.matmul,
                                       operation=Op.matmul)

    def conv_2D(self, mask, right, down):

        return Node.operation_template(a=self, b=mask, func=lambda x, c: conv2D_func(x, c, right, down),
                                       operation=Op.conv_2D, optional_info=[right, down])

    def conv_3D(self, mask, right, down, back):

        return Node.operation_template(a=self, b=mask, func=lambda x,c: conv3D_func(x, c, right, down, back),
                                       operation=Op.conv_3D,
                                       optional_info=[right, down, back])

    def concat_vectors(self, other):

        return Node.operation_template(a=self, b=other, func=lambda x, y: np.concatenate((x, y), axis=0),
                                       operation=Op.concat_vectors)

    # unary operations:

    def __neg__(self):

        return Node.operation_template(a=self, func=np.ndarray.__neg__, operation=Op.neg)

    def exp(self):

        return Node.operation_template(a=self, func=lambda x: np.exp(np.clip(x, -25, 25)), operation=Op.exp)

    def log(self):

        return Node.operation_template(a=self, func=lambda x: np.log(x + 1e-8), operation=Op.log)

    def sum(self):

        res = Node.operation_template(a=self, func=np.ndarray.sum, operation=Op.sum)
        res.value = np.reshape(res.value, newshape=(1, ))

        return res

    def reshape(self, newshape):

        res = Node.operation_template(a=self, func=lambda x: np.reshape(x, newshape), operation=Op.reshape,
                                      optional_info=[self.value.shape])

        return res

    def duplicate(self, shape):

        if self.value.shape != (1, ):
            print("duplicate works only on nodes with a single value.")
            exit(1)

        return Node.operation_template(a=self, func=(lambda val: val * np.ones(shape)), operation=Op.dup)

    def abs(self):

        return Node.operation_template(a=self,
                                       func=np.absolute,
                                       operation=Op.abs)

    def sin(self):

        return Node.operation_template(a=self,
                                       func=np.sin,
                                       operation=Op.sin)


    def max_pooling1D(self, mask_shape, down):

        return Node.operation_template(a=self,
                                       func=lambda x: max_pooling1D_func(x, mask_shape, down),
                                       operation=Op.maxpool_1D,
                                       optional_info=[mask_shape[0], down])

    def max_pooling2D(self, mask_shape, right, down):

        return Node.operation_template(a=self,
                                       func=lambda x: max_pooling2D_func(x, mask_shape, right, down),
                                       operation=Op.maxpool_2D,
                                       optional_info=[mask_shape[0], mask_shape[1], right, down])

    def max_pooling3D(self, mask_shape, right, down, back):

        return Node.operation_template(a=self,
                                       func=lambda x: max_pooling3D_func(x, mask_shape, right=right, down=down, back=back),
                                       operation=Op.maxpool_3D,
                                       optional_info=[mask_shape[0], mask_shape[1], mask_shape[2], right, down, back])

    # Activations (for optimizing running time):

    def relu(self):

        return Node.operation_template(a=self, func=activate.relu, operation=Op.relu)

    def tanh(self):

        return Node.operation_template(a=self, func=activate.tanh, operation=Op.tanh)


    def sigmoid(self):

        return Node.operation_template(a=self, func=activate.sigmoid, operation=Op.sigmoid)

    def softmax(self):

        return Node.operation_template(a=self, func=activate.softmax, operation=Op.softmax)


    #  gradient computation function:

    def grad(self, grad_dict=None):

        """
        calculates the gradient of the calling node according to the parameters at the computation graph
        (node should be a function with single-dimension-output).
        if grad_dict is None, returns a dictionary containing the calculated values.
        if grad_dict is given (not None), adds the gradient to the dictionary.
        """

        # if (type(self.value) not in [int, float]) and (self.value.shape not in [(1, 1), (1, )]):
        #     print("The given function has multiple-outputs. \n"
        #           "The input should be a uni-output function.")
        #     exit(1)

        if grad_dict is None:
            dict_given = False
            grad_dict = {}
        else:
            dict_given = True

        topo_order_list = self.__topological_sort__()

        self.diff = 1

        for i in range(1, len(topo_order_list)):

            node = topo_order_list[i]

            if (node.is_constant or node.parents is None) and not node.is_param:
                # no need to calculate ∂self/∂node.
                continue

            for child in node.childs:
                if child.is_relevant and (not child.is_constant):
                    if node.diff is None:
                        node.diff = child.__simple_grad__(node)
                    else:
                        node.diff += child.__simple_grad__(node)

            if node.is_param:
                if dict_given:
                    grad_dict[node] += node.diff
                else:
                    # dictionary was not given.
                    grad_dict[node] = node.diff

        for node in topo_order_list:
            if ~node.is_param:
                # throw its 'diff' field (for memory efficiency)
                node.diff = None

        return grad_dict

    def __simple_grad__(child, parent):

        """
        assuming other is a self's parent, gets child and parent nodes in the computation graph,
        and also ∂ancestor/∂child, and returns np.matmul(∂ancestor/∂child, ∂child/∂parent),
        while computing it efficiently, using the "BasicGradients" python module functionality.
        """

        if child.is_constant:
            return 0

        if len(child.parents) == 2:
            ind = parent.__is_first_arg_of__(child)

            x1 = child.parents[0].value
            x2 = child.parents[1].value

            func = bg.jvp_dict[child.operation][~ind]

            res = func(child.diff, x1, x2, y=child.value, additional_info=child.optional_info)

        else:
            # len(child.parents) == 1;

            res = bg.jvp_dict[child.operation](child.diff,
                                               x=parent.value,
                                               y=child.value,
                                               additional_info=child.optional_info
                                               )

        return res
