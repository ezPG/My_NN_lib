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
    

# class Conv2D(BaseLayer):
#     def __init__(self, in_channels, out_channels, kernel_size, 
#                 stride=1, padding=0, dilation=1, 
#                 groups=1, bias=True, padding_mode='zeros'):

#         self.in_channels = in_channels
#         self.out_channels = out_channels
        
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.stride = stride if isinstance(stride, tuple) else (stride, stride)
#         self.padding = padding if isinstance(padding, tuple) else (padding, padding)
#         self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
#         self.groups = groups
#         self.bias = bias
#         self.padding_mode = padding_mode
        

#         fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
#         scale = 2 / fan_in  
#         self.kernel = np.random.randn(out_channels, in_channels // groups, *self.kernel_size) * np.sqrt(scale) 
#         self.b = np.random.randn(out_channels, 1) if bias else None
        
#     def forward(self, X):
#         self.X = X  
#         self.out_shape = (
#             (X.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1,
#             (X.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
#         )
        
#         # Pad the input if needed
#         if self.padding != (0, 0):
#             X = np.pad(X, ((0, 0), (0, 0), self.padding, self.padding), mode=self.padding_mode)

#         output = np.zeros((X.shape[0], self.out_channels, *self.out_shape))
#         for i in range(self.out_shape[0]):
#             for j in range(self.out_shape[1]):
#                 x_slice = X[:, :, i*self.stride[0]:i*self.stride[0]+self.kernel_size[0], j*self.stride[1]:j*self.stride[1]+self.kernel_size[1]]
#                 output[:, :, i, j] = np.sum(x_slice[:, np.newaxis, :, :, :] * self.W[np.newaxis, :, :, :, :], axis=(2, 3, 4))

#         if self.bias is not None:
#             output += self.b.reshape(1, -1, 1, 1)

#         return output
    
#     def backward(self, output_grad, lr):
#         return


class Conv2D(BaseLayer):
    def __init__(self, in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, 
                groups=1, bias=True, padding_mode='zeros'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Dilation determines the spacing between kernel elements. 
        # A dilation rate of 1 means normal convolution, while a higher rate means a larger receptive field.
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        # Groups allow for grouped convolutions, where input channels are split into groups 
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        
        # fan_in is the number of input units to the layer, calculated as the product of in_channels and the kernel size.
        # It is used to scale the initial random weights to keep the variance of activations consistent across layers.
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        scale = 2 / fan_in  
        self.kernels = np.random.randn(out_channels, in_channels // groups, *self.kernel_size) * np.sqrt(scale) 
        self.biases = np.random.randn(out_channels, 1, 1) if bias else None
        
    def forward(self, X):
        self.X = X
        batch_size, _, input_height, input_width = X.shape
        
        self.output_height = (input_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        self.output_width = (input_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        # Pad the input if needed
        if self.padding != (0, 0):
            X = np.pad(X, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), mode=self.padding_mode)
        
        output = np.zeros((batch_size, self.out_channels, self.output_height, self.output_width))
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                for k in range(batch_size):
                    output[k, i] += signal.correlate2d(X[k, j], self.kernels[i, j], mode='valid')[::self.stride[0], ::self.stride[1]]

        if self.bias:
            output += self.biases

        self.output = output
        return output
    
    def backward(self, output_grad, lr):
        batch_size, _, output_height, output_width = output_grad.shape
        _, _, input_height, input_width = self.X.shape
        
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.X)
        
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                for k in range(batch_size):
                    kernels_gradient[i, j] += signal.correlate2d(self.X[k, j], output_grad[k, i], mode='valid')
                    input_gradient[k, j] += signal.convolve2d(output_grad[k, i], self.kernels[i, j], mode='full')
        
        self.kernels -= lr * kernels_gradient
        if self.bias:
            self.biases -= lr * np.sum(output_grad, axis=(0, 2, 3)).reshape(self.biases.shape)

        return input_gradient


class MaxPooling2D(BaseLayer):
    def __init__(self, pool_size=(2, 2), stride=None, padding=(0, 0)):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
    
    def forward(self, X):
        self.X = X
        batch_size, channels, input_height, input_width = X.shape
        
        self.output_height = (input_height + 2 * self.padding[0] - self.pool_size[0]) // self.stride[0] + 1
        self.output_width = (input_width + 2 * self.padding[1] - self.pool_size[1]) // self.stride[1] + 1
        
        if self.padding != (0, 0):
            X = np.pad(X, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), mode='constant')
        
        output = np.zeros((batch_size, channels, self.output_height, self.output_width))
        self.max_indices = np.zeros_like(output, dtype=int)
        
        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]
                
                X_slice = X[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(X_slice, axis=(2, 3))
                
                max_indices = np.argmax(X_slice.reshape(batch_size, channels, -1), axis=2)
                self.max_indices[:, :, i, j] = max_indices
        
        return output
    
    def backward(self, output_grad, lr):
        input_grad = np.zeros_like(self.X)
        
        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]
                
                grad_slice = input_grad[:, :, h_start:h_end, w_start:w_end]
                max_indices = self.max_indices[:, :, i, j]
                
                for batch in range(output_grad.shape[0]):
                    for channel in range(output_grad.shape[1]):
                        h_idx, w_idx = np.unravel_index(max_indices[batch, channel], self.pool_size)
                        grad_slice[batch, channel, h_idx, w_idx] += output_grad[batch, channel, i, j]
        
        return input_grad
