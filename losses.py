import numpy as np
from module import Module

class Loss(Module):
    def forward(self, predictions, targets):
        raise NotImplementedError("Forward method should be implemented in subclasses")
    
    def backward(self):
        raise NotImplementedError("Backward method should be implemented in subclasses")
    
class MSELoss(Loss):
    def forward(self, predictions, targets):
        self.cache = (predictions, targets)
        return np.mean((predictions - targets) ** 2)
    
    def backward(self):
        predictions, targets = self.cache
        return 2 * (predictions - targets) / targets.size
    
class CrossEntropyLoss(Loss):
    def forward(self, predictions, targets):
        exp_preds = np.exp(predictions - np.max(predictions, axis = 1, keepdims=True))
        softmax_preds = exp_preds / np.sum(exp_preds, axis = 1, keepdims=True)

        self.cache = (softmax_preds, targets)

        log_likelihood = -np.log(softmax_preds[np.arange(len(targets)), targets])
        return np.mean(log_likelihood)
    
    def backward(self):
        softmax_preds, targets = self.cache
        # grad = softmax_preds - targets
        grad = softmax_preds.copy()
        grad[np.arange(len(targets)), targets] -= 1
        return grad / len(targets)
    
class BCEWithLogitsLoss(Loss):
    def forward(self, predictions, targets):
        sigmoid_preds = 1 / (1 + np.exp(-predictions))

        # cache sigmoid output for the backprop
        self.cache = (sigmoid_preds, targets)
        # compute BCE loss
        bce_loss = - (targets * np.log(sigmoid_preds) + (1 - targets) * np.log(1 - sigmoid_preds))
        return np.mean(bce_loss)

    def backward(self):
        sigmoid_preds, targets = self.cache
        grad =  (sigmoid_preds - targets) 

        return grad / targets.size