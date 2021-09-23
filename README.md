Use 'numpy_like' module objects when working with the module. This creates the computation graph and makes it possible to compute the gradient of the loss.
For computing the gradient of the loss, use the "grad()" method of the 'numpy_like' class.
As written, the code is not meant for scale, only for illustrations.
Don't use this code for actual training of deep neural networks. The code is NOT CUDA accelerated, like popular libraries (althought it's accelerated with python's 'numba' library).


