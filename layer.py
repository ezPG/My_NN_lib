import numpy as np
from scipy import signal

class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        return
    
    def backward(self, output_grad, lr):
        return


class Linear(BaseLayer):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features)
        self.b = np.random.randn(out_features, 1)
    
    def forward(self, X):
        self.X = X
        return np.dot(self.W * self.X) + self.b

    def backward(self, output_grad, lr):
        W_grads = np.dot(output_grad, self.X.T) #dE/dW = dE/dY * X.t
        self.W -= lr * W_grads  #Update Weights wrt the gradient
        self.b -= lr * output_grad  #dE/dY
        
        return np.dot(self.W.T, output_grad) #dE/dX = W.t * dE/dY
    

class Conv2D(BaseLayer):
    def __init__(self, in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, 
                groups=1, bias=True, padding_mode='zeros'):

        self.in_channels = in_channels
        self.out_channels = out_channels