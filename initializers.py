# Essentials
import numpy as np

# Xavier initializer for tanh, sigmoid
def xavier(dim0, dim1):
	return np.random.uniform(-1, 1, (dim0, dim1)) *  np.sqrt(6.0 / (dim0 + dim1))

# He initializer for Relu
def he(dim0, dim1):
	return np.random.uniform(-1, 1, (dim0, dim1)) *  np.sqrt(2.0 / dim0)

