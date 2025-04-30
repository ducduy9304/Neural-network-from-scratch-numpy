import numpy as np
from module import Module


class Activation(Module):
    def __init__(self, activation_name):
        super().__init__()
        self.activation_functions = {
            'tanh': (self.tanh, self.tanh_prime),
            'sigmoid': (self.sigmoid, self.sigmoid_prime),
            'relu': (self.relu, self.relu_prime),
            'leakyrelu': (self.leakyrelu, self.leakyrelu_prime),
            'softmax': (self.softmax, self.softmax_prime),
        }

        self.activation_name = activation_name.lower()
        if self.activation_name not in self.activation_functions:
            raise ValueError(f"Unsupported activation function: {self.activation_name}. Supported functions: {list(self.activation_functions.keys())}")
        
    def forward(self, x):
        self.x = x # store input for backprop
        activation_func, _ = self.activation_functions[self.activation_name]
        return activation_func(self.x)
    
    def backward(self, grad_out):
        if self.activation_name == "softmax":
            # softmax is typically used with cross-entropy loss, so we return the gradient as-is
            return grad_out
        else:
            _, gradient_func = self.activation_functions[self.activation_name]
            return grad_out * gradient_func(self.x)
        
    
    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1/(1+np.exp(-x))
    
    @staticmethod
    def sigmoid_prime(x):
        s = Activation.sigmoid(x)
        return s*(1-s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_prime(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_prime(x):
        return np.where(x>0, 1, 0)
    
    @staticmethod
    def leakyrelu(x, alpha = 0.01):
        return np.where(x>0, x, alpha*x)
    
    @staticmethod
    def leakyrelu_prime(x, alpha = 0.01):
        return np.where(x>0, 1, alpha)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_prime(x):
        # Softmax derivative is generally not needed separately when used with cross-entropy
        raise NotImplementedError("Softmax derivative is rarely used directly in practice.")