from nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Tensor


class LeNet(object):
    def __init__(self):
        self.layers = [
            Conv2d(1, 6, 5, padding=2),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(6, 16, 5),
            ReLU(),
            MaxPool2d(2, 2),
            Flatten(),
            Linear(400, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            Linear(84, 10)
        ]

    def __call__(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                params.append(layer.weight)
            if hasattr(layer, 'bias'):
                params.append(layer.bias)
        return params


if __name__ == '__main__':
    import numpy as np
    x = Tensor(np.random.randn(512, 32, 32, 1))
    net = LeNet()
    y = net(x)
    print(x.shape, y.shape)

    grad = np.random.randn(*y.shape)
    y.backward(grad)
    print(x.grad.sum())