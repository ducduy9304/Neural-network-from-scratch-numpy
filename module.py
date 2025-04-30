import numpy as np
import os


class Parameter:
    def __init__(self, data):
        self.data = data  # The actual parameter values
        self.grad = np.zeros_like(data)  # The gradient of the parameter

    def zero_grad(self):
        self.grad.fill(0)



class Module():
    def __init__(self):
        self.training = True
        self.params = {}

    def forward(self, *inputs):
        # inputs is a tuple: (input1, input2, ...)
        raise NotImplementedError("Forward must implement this method")
    
    def backward(self, *outputs):
        raise NotImplementedError("Backward must implement this method")
    
    def add_param(self, name, value):
        """Add a parameter and initialize its gradient"""
        self.params[name] = Parameter(value)

    
    def zero_grad(self):
        """Set all gradients to zero before a new backprop"""
        for param in self.params.values():
            param.zero_grad()

    def parameters(self):
        for param in self.params.values():
            yield param
        # If the module contains submodules, yield their parameters as well
        for module in getattr(self, 'modules', []):
            yield from module.parameters()

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

    def __call__(self, *inputs):
        # This will allow us to call the module like a function and invoke forward()
        return self.forward(*inputs)
    


    def save_model(self, filepath):
        """
        Save the model parameters to a file using NumPy.
        """
        model_data = {
            'params': {name: param.data for name, param in self.params.items()},
            'gradients': {name: param.grad for name, param in self.params.items()},
            'state': self.training  # Save whether the model is in training mode
        }
        # Convert dictionaries to NumPy-compatible format
        np.savez(filepath, 
                params=np.array(list(model_data['params'].items()), dtype=object), 
                gradients=np.array(list(model_data['gradients'].items()), dtype=object), 
                state=model_data['state'])
        print(f"Model saved successfully into {filepath}")



    def load_model(self, filepath):
        """
        Load the model parameters from a file using NumPy.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file '{filepath}' does not exist.")
        
        # Load data from the .npz file
        model_data = np.load(filepath, allow_pickle=True)
        params = dict(model_data['params'].tolist())
        gradients = dict(model_data['gradients'].tolist())
        self.training = model_data['state'].item()

        # Update parameters and gradients in the model
        for name, param in self.params.items():
            if name in params:
                param.data = params[name]
                param.grad = gradients.get(name, None)  # Gradients may not exist for all parameters
        
        print(f"Model loaded successfully from {filepath}")



class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        self.x = x
        for module in self.modules:
            x = module(x)
        return x
    
    def backward(self, grad_out):
        for module in reversed(self.modules):
            grad_out = module.backward(grad_out)
        return grad_out
    


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.add_param('W', np.random.randn(in_features, out_features) * np.sqrt(2. / in_features))
        self.add_param('b', np.zeros(out_features))
        self.cache = None # store input for backprop

    def forward(self, x):
        self.cache = x
        W = self.params['W'].data
        b = self.params['b'].data
        return x @ W + b
    
    def backward(self, grad_out):
        x = self.cache
        self.params['W'].grad += x.T @ grad_out
        self.params['b'].grad += np.sum(grad_out, axis=0)
        dx = grad_out @ self.params['W'].data.T
        return dx