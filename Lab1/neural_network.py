import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, A):
        self.mask = (A <= 0)
        Z = A.copy()
        Z[self.mask] = 0
        return Z

    def backward(self, dZ):
        # dZ is the gradient from the next layer
        dA = dZ.copy()
        dA[self.mask] = 0
        return dA

class Sigmoid:
    def __init__(self):
        self.Z = None

    def forward(self, A):
        self.Z = 1.0 / (1.0 + np.exp(-A))
        return self.Z

    def backward(self, dZ):
        local_gradient = self.Z * (1.0 - self.Z)
        dA = dZ * local_gradient  # upstream gradient * local gradient
        return dA

class Tanh:
    def __init__(self):
        self.Z = None

    def forward(self, A):
        self.Z = np.tanh(A)
        return self.Z

    def backward(self, dZ):
        local_gradient = 1.0 - self.Z ** 2
        dA = dZ * local_gradient
        return dA

class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim) # weight matrix in (d, m) shape
        self.b = np.zeros((1, output_dim)) # bias vector in (1, m) shape
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X # Store input for backward pass
        self.A = X @ self.W + self.b
        return self.A

    def backward(self, dA):
        self.dW += self.X.T @ dA # gradient w.r.t. weights
        self.db += np.sum(dA, axis=0, keepdims=True) # gradient w.r.t. biases
        dX = dA @ self.W.T # gradient w.r.t. inputs
        return dX

class NeuralNetwork:
    def __init__(self, h1_dim=4, h2_dim=4):
        self.layers = [
            Linear(2, h1_dim),
            Tanh(),
            Linear(h1_dim, h2_dim),
            ReLU(),
            Linear(h2_dim, 1),
            Sigmoid()
        ]
        self.parameters = [layer for layer in self.layers if isinstance(layer, Linear)]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, loss_grad):
        dout = loss_grad
        for layer in reversed(self.layers):
            dout = layer.backward(dout)