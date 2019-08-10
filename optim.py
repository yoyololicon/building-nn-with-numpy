import numpy as np


class SGD(object):
    def __init__(self, params: list, lr: float, mommentum=0, weight_decay=0):
        self.params = params
        self.lr = lr
        self.momentum = mommentum
        self.weight_decay = weight_decay


        if self.momentum:
            self.state = {}
            for p in self.params:
                self.state[id(p)] = np.zeros_like(p)

    def zero_grad(self):
        for p in self.params:
            p.grad[:] = 0

    def step(self):
        for p in self.params:
            d_p = p.grad
            if self.momentum:
                self.state[id(p)] *= self.momentum
                self.state[id(p)] += d_p
                d_p = self.state[id(p)]
            if self.weight_decay:
                d_p = d_p + self.weight_decay * p
            p -= self.lr * d_p
