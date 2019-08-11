import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import sparse


class Tensor(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.grad = np.zeros(input_array.shape)
        obj.from_where = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.grad = getattr(obj, 'grad', np.zeros(obj.shape))
        self.from_where = getattr(obj, 'from_where', None)

    def backward(self, grad: np.array):
        if grad.shape != self.grad.shape:
            grad = grad.reshape(*self.grad.shape)
        self.grad += grad
        if self.from_where is not None:
            self.from_where.backward()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Tensor):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, Tensor):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super(Tensor, self).__array_ufunc__(ufunc, method,
                                                      *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(Tensor)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results


def _image_zero_pad_or_crop(images, pad_tup):
    pad_tup = [*pad_tup]
    if pad_tup[0] < 0:
        images = images[:, :, -pad_tup[0]:pad_tup[0]]
        pad_tup[0] = 0
    if pad_tup[1] < 0:
        images = images[..., -pad_tup[1]:pad_tup[1]]
        pad_tup[1] = 0
    if pad_tup[0] == 0 and pad_tup[1] == 0:
        return images
    pad_tup = ((0,) * 2,) * 2 + ((pad_tup[0],) * 2, (pad_tup[1],) * 2)
    return np.pad(images, pad_tup, 'constant', constant_values=0)


def _make_pair(x):
    if type(x) == tuple:
        return x
    else:
        return (x,) * 2


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, output_shape=None):
    s0, s1 = _make_pair(stride)
    d0, d1 = _make_pair(dilation)
    padding = _make_pair(padding)
    *_, k0, k1 = weight.shape

    x = _image_zero_pad_or_crop(x, padding)

    if output_shape:
        new_shape = x.shape[:2] + output_shape + (k0, k1)
    else:
        new_shape = x.shape[:2] + _conv_shape(x.shape[2:], (k0, k1), stride, 0, dilation) + (k0, k1)

    new_stride = x.strides[:2] + \
                 (x.strides[2] * s0, x.strides[3] * s1,
                  x.strides[2] * d0, x.strides[3] * d1)
    x = as_strided(x, new_shape, new_stride, True, False)
    y = np.rollaxis(np.tensordot(x, weight, axes=([1, 4, 5], [1, 2, 3])), 3, 1)
    return y if bias is None else y + bias[:, None, None]


def transpose_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, output_padding=0):
    s0, s1 = _make_pair(stride)
    d0, d1 = _make_pair(dilation)
    padding = _make_pair(padding)
    *_, k0, k1 = weight.shape

    if s0 > 1:
        zero_x = np.zeros_like(x[:, :, :-1]).repeat(s0 - 1, 2)
        x = np.insert(zero_x, np.arange(x.shape[2]) * (s0 - 1), x, 2)
    if s1 > 1:
        zero_x = np.zeros_like(x[..., :-1]).repeat(s1 - 1, 3)
        x = np.insert(zero_x, np.arange(x.shape[3]) * (s1 - 1), x, 3)

    if output_padding:
        output_padding = _make_pair(output_padding)
        x = np.pad(x, ((0,) * 2,) * 2 + ((0, output_padding[0]), (0, output_padding[1])), 'constant',
                   constant_values=0)

    weight = np.flip(weight, (2, 3)).swapaxes(0, 1)
    y = conv2d(x, weight, bias,
               padding=((k0 - 1) * d0 - padding[0], (k1 - 1) * d1 - padding[1]), dilation=dilation)

    return y


def _conv_shape(shape, k, s, p, d):
    s = _make_pair(s)
    d = _make_pair(d)
    p = _make_pair(p)
    new_H = (shape[0] + 2 * p[0] - d[0] * (k[0] - 1) - 1) // s[0] + 1
    new_W = (shape[1] + 2 * p[1] - d[1] * (k[1] - 1) - 1) // s[1] + 1
    return (new_H, new_W)


def _transposeconv_shape(shape, k, s, p, d):
    s = _make_pair(s)
    d = _make_pair(d)
    p = _make_pair(p)
    new_H = (shape[0] - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + 1
    new_W = (shape[1] - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + 1
    return (new_H, new_W)


class _module(object):
    def __call__(self, x: Tensor):
        self.input_ref = x
        y = self.forward(x)
        y.from_where = self
        self.output_ref = y
        return y

    def forward(self, x: Tensor):
        raise NotImplementedError

    def backward(self):
        self.input_ref.backward(self.grad_pass(self.output_ref.grad))

    def grad_pass(self, y_grad: np.array):
        raise NotImplementedError


class ReLU(_module):
    def forward(self, x: Tensor):
        self.idx = np.where(x > 0)
        return np.maximum(0, x)

    def grad_pass(self, y_grad: np.array):
        x_grad = np.zeros_like(y_grad)
        x_grad[self.idx] = y_grad[self.idx]
        return x_grad


class Linear(_module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Tensor(np.random.randn(out_channels, in_channels) / np.sqrt(in_channels * 0.5))
        self.bias = Tensor(np.random.uniform(-1 / np.sqrt(in_channels), 1 / np.sqrt(in_channels), out_channels))

    def forward(self, x: Tensor):
        return x @ self.weight.T + self.bias

    def grad_pass(self, y_grad: np.array):
        w_grad = y_grad.T @ self.input_ref
        b_grad = y_grad.sum(0)
        self.weight.backward(w_grad)
        self.bias.backward(b_grad)

        x_grad = y_grad @ self.weight
        return x_grad


class MaxPool2d(_module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        self.kernel_size = _make_pair(kernel_size)
        self.padding = _make_pair(padding)
        self.stride = _make_pair(stride)
        self.dilation = _make_pair(dilation)

    def forward(self, x: Tensor):
        x = _image_zero_pad_or_crop(x, self.padding)

        input_size = np.prod(x.shape)
        flat_idx = np.arange(input_size).reshape(*x.shape)
        d0, d1 = self.dilation
        k0, k1 = self.kernel_size
        s0, s1 = self.stride
        new_shape = x.shape[:2] + _conv_shape(x.shape[2:], self.kernel_size, self.stride, 0,
                                              self.dilation) + self.kernel_size

        new_stride = x.strides[:2] + \
                     (x.strides[2] * s0, x.strides[3] * s1, x.strides[2] * d0, x.strides[3] * d1)
        unfolded_x = as_strided(x, new_shape, new_stride, True, False)

        unfolded_x = unfolded_x.reshape(*unfolded_x.shape[:4], -1)

        unfolded_flat_idx = as_strided(flat_idx, new_shape, new_stride, True, False)
        unfolded_flat_idx = unfolded_flat_idx.reshape(*unfolded_x.shape)

        max_idx = np.argmax(unfolded_x, 4)
        matrix_idx = np.take_along_axis(unfolded_flat_idx, max_idx[..., None], 4).squeeze(4).ravel()

        ouput_size = np.prod(max_idx.shape)
        sp_matrx = sparse.coo_matrix((np.broadcast_to(1, ouput_size), (matrix_idx, np.arange(ouput_size))),
                                     shape=(input_size, ouput_size))
        y = x.ravel() @ sp_matrx
        self.sp_matrix = sp_matrx
        self.x_view = x.shape
        return y.reshape(*max_idx.shape).view(Tensor)

    def grad_pass(self, y_grad: np.array):
        x_grad = y_grad.ravel() @ self.sp_matrix.T
        x_grad = x_grad.reshape(*self.x_view)
        if self.padding[0]:
            x_grad = x_grad[:, :, self.padding[0]:-self.padding[0]]
        if self.padding[1]:
            x_grad = x_grad[..., self.padding[1]:-self.padding[1]]
        return x_grad


class Conv2d(_module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        self.kernel_size = _make_pair(kernel_size)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.weight = Tensor(np.random.randn(out_channels, in_channels, *self.kernel_size) / np.sqrt(
            0.5 * np.prod(self.kernel_size) * in_channels))
        bound = np.power(in_channels * np.prod(self.kernel_size), -0.5)
        self.bias = Tensor(np.random.uniform(-bound, bound, out_channels))

    def forward(self, x: Tensor):
        y = conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
        return y.view(Tensor)

    def grad_pass(self, y_grad: np.array):
        x = self.input_ref.swapaxes(0, 1)
        grad_weight = y_grad.swapaxes(0, 1)

        w_grad = conv2d(x, grad_weight, stride=self.dilation, padding=self.padding, dilation=self.stride,
                        output_shape=self.kernel_size).swapaxes(0, 1)
        self.weight.backward(w_grad)
        b_grad = y_grad.sum((0, 2, 3))
        self.bias.backward(b_grad)

        new_shape = _transposeconv_shape(y_grad.shape[-2:], self.kernel_size, self.stride, self.padding, self.dilation)
        output_padding = (self.input_ref.shape[2] - new_shape[0], self.input_ref.shape[3] - new_shape[1])

        x_grad = transpose_conv2d(y_grad, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation,
                                 output_padding=output_padding)
        return x_grad


class Flatten(_module):
    def forward(self, x: Tensor):
        self.x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        return Tensor(x)

    def grad_pass(self, y_grad: np.array):
        return y_grad.reshape(*self.x_shape)


def cross_entropy_loss(logits, target):
    # combine softmax, loss, and gradient
    batch = logits.shape[0]
    max_logits = logits.max(1, keepdims=True)
    logits = logits - max_logits
    logsoftmax = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    loss = -np.take_along_axis(logsoftmax, target[:, None], 1).sum()
    grad = np.exp(logsoftmax)
    grad[np.arange(batch), target] -= 1
    loss /= batch
    grad /= batch
    return loss, grad


if __name__ == '__main__':
    n = MaxPool2d(1, stride=2, padding=1)
    n2 = Conv2d(3, 16, 3, padding=1)
    n3 = ReLU()

    x = Tensor(np.random.randn(1, 3, 64, 64))
    y = n(n3(n2(x)))
    print(y.shape)
    l = Flatten()
    y = l(y)
    # print(y.grad.shape, type(y))
    n4 = Linear(y.shape[1], 10)
    y = n4(y)
    print(y.shape)
    y.backward(np.random.randn(*y.shape))
    # print(x.grad[0, ..., 0])
    print(x.grad[0, ..., 0], x[0, ..., 0])
    print(n2.weight.grad.sum(), n2.bias.grad.sum())
