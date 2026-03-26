import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.W -= self.lr * param.dW
            param.b -= self.lr * param.db

    def zero_grad(self):
        for param in self.parameters:
            param.dW.fill(0)
            param.db.fill(0)

class StepLR:
    def __init__(self, optimizer, step_size=10000, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma

    def step(self, current_epoch):
        if current_epoch > 0 and current_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma