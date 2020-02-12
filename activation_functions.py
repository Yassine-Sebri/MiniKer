# Essentials
import numpy as np

# Defining activation functions with their derivatives
# For Relu
def relu(X):
    return X * (X > 0)
def relu_derivative(X):
    return X > 0

# For sigmoid
def sigmoid(X):
    return 1 / (1 + np.exp(-X))
def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))

# For tanh
def tanh(X):
    return np.tanh(X)
def tanh_derivative(X):
    return 1 - np.power(np.tanh(X), 2)
