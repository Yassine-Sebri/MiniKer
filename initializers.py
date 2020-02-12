# Essentials
import numpy as np

def xavier(dim0, dim1):
	return np.random.uniform(-1, 1, (dim0, dim1)) *  np.sqrt(6.0 / 513)


