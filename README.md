# Auto Differentiation

An auto-differentiation module for neural networks. Illustrates what's going on under the hood of popular deep-learning libraries like Tensorflow and Pytorch.

Use 'numpy_like' module objects when working with the module. This creates the computation graph and makes it possible to compute the gradient of the loss.
For computing the gradient of the loss, use the "grad()" method of the 'numpy_like' class.
Note that this code is not optimized for scaling, and meant for illustrating what is going on at the core of popular deep-learning libraries.


