import numpy as np
from .Layer import Layer

class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = (np.random.rand(input_dim, output_dim) - 0.5) * 0.01
        self.bias = np.random.rand(1, output_dim) - 0.5

    def forward_propagation(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output

    def backward_propagation(self, next_layer_grads, lr):
        grads = np.dot(next_layer_grads, self.weights.T)
        dW = np.dot(self.input.T, next_layer_grads)
        db = np.sum(next_layer_grads, axis=0)
        
        self.weights -= lr * dW
        self.bias -= lr * db
        return grads
    
class Activation(Layer):
    def __init__(self, activation):
        self.activation = activation
        if (activation == 'tanh'):
            self.__forward_prop_fn__ = self.__tanh__
            self.__backward_prop_fn__ = self.__dtanh__
        elif (activation == 'sigmoid'):
            self.__forward_prop_fn__ = self.__sigmoid__
            self.__backward_prop_fn__ = self.__dsigmoid__
        elif (activation == 'relu'):
            self.__forward_prop_fn__ = self.__relu__
            self.__backward_prop_fn__ = self.__drelu__

    def forward_propagation(self, x):
        self.input = x
        self.output = self.__forward_prop_fn__(x)
        return self.output

    def backward_propagation(self, next_layer_grads, learning_rate):
        return self.__backward_prop_fn__(self.input) * next_layer_grads
    
    def __tanh__(self, z):
        return np.tanh(z)

    def __dtanh__(self, z):
        return 1-np.tanh(z)**2

    def __sigmoid__(self, z):
        return (1 / (1 + np.exp(-z) + 10e-8))

    def __dsigmoid__(self, z):
        s = self.__sigmoid__(z)
        return s * (1 - s)
    
    def __relu__(self, z):
        return np.maximum(z, 0)
    
    def __drelu__(self, z):
        dz = np.ones(z.shape)
        dz[self.input <= 0] = 0
        return dz