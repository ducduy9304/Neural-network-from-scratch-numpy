import numpy as np
from module import Module

class Optimizer:
    def __init__(self, parameters, lr=0.01, momentum=0):
        self.parameters = list(parameters)  # List of Parameter objects
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(param.data) for param in self.parameters]
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
    
    def step(self):
        raise NotImplementedError("Subclasses should implement this")

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0):
        super().__init__(parameters, lr, momentum)

    def step(self):
        for idx, param in enumerate(self.parameters):
            grad = param.grad
            velocity = self.momentum * self.velocities[idx] - self.lr * grad
            param.data += velocity
            self.velocities[idx] = velocity


class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step

        # Initialize first and second moment vectors for each parameter
        self.m = [np.zeros_like(param.data) for param in self.parameters]  # First moment
        self.v = [np.zeros_like(param.data) for param in self.parameters]  # Second moment

    def step(self):
        """
        Perform a single optimization step.
        """
        self.t += 1  # Increment time step
        for idx, param in enumerate(self.parameters):
            grad = param.grad

            # Update biased first moment estimate
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
