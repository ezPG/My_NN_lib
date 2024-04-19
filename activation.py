import numpy as np
from layer import BaseLayer

class Activation(BaseLayer):
    def __init__(self, activation, activation_prime):
        self.A = activation
        self.dA = activation_prime
    
    def forward(self, X):
        self.X = X
        return self.A(self.X)

    def backward(self, output_grad, lr):
        return np.multiply(output_grad, self.dA(self.X))
    

class Softmax(BaseLayer):
    def __init__(self):
        pass
    
    def forward(self, X):
        self.out = np.exp(X)/np.sum(np.exp(X))
        return self.out
    
    def backward(self, output_grad, lr):
        n = np.size(self.out)
        M = np.tile(self.out, n) #output vector replicated n times
        return np.dot(M * (np.identity(n) -  M.T), output_grad)


class ReLU(Activation):
    def __init__(self):
        relu = self.relu()
        relu_prime = self.relu_prime()
        super().__init__(relu, relu_prime)

    def relu(self, X):
        return np.maximum(0, X)

    def relu_prime(self, X):
        return np.where(X > 0, 1, 0)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = self.sigmoid()
        sigmoid_prime = self.sigmoid_prime()
        super().__init__(sigmoid, sigmoid_prime)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_prime(self, X):
        s = self.sigmoid(X)
        return s * (1 - s)

class Tanh(Activation):
    def __init__(self):
        tanh = self.tanh()
        tanh_prime = self.tanh_prime()

        super().__init__(tanh, tanh_prime)
        
    def tanh(self, X):
        return np.tanh(X)
    
    def tanh_prime(self, X):
        return 1 - np.square(self.tanh(X))
    