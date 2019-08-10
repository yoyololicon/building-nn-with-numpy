from nn import *
import torch
from torch import nn
import torch.nn.functional as F
import pytest

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize('k_size', list(range(1, 6)))
@pytest.mark.parametrize('p', list(range(3)))
@pytest.mark.parametrize('s', list(range(1, 4)))
@pytest.mark.parametrize('d', list(range(1, 4)))
def test_conv2d(k_size, s, p, d):
    input = np.random.rand(64, 3, 64, 64)
    out_size = 16

    x1 = torch.Tensor(input)
    x2 = Tensor(input)
    x1.requires_grad = True

    n1 = nn.Conv2d(3, out_size, k_size, s, p, d)
    n2 = Conv2d(3, out_size, k_size, s, p, d)
    n2.weight[:] = n1.weight.data.numpy()
    n2.bias[:] = n1.bias.data.numpy()

    y1 = n1(x1)
    y2 = n2(x2)

    assert y1.shape == y2.shape
    assert np.allclose(y1.detach().numpy(), y2)

    loss = y1.sum()
    loss.backward()

    y2.backward(np.ones(y2.shape))

    assert np.allclose(x1.grad.numpy(), x2.grad)
    assert np.allclose(n1.weight.grad.numpy(), n2.weight.grad)
    assert np.allclose(n1.bias.grad.numpy(), n2.bias.grad)


@pytest.mark.parametrize('k_size', list(range(2, 5)))
@pytest.mark.parametrize('p', list(range(2)))
@pytest.mark.parametrize('s', list(range(1, 4)))
@pytest.mark.parametrize('d', list(range(1, 4)))
def test_maxpool2d(k_size, s, p, d):
    input = np.random.rand(64, 3, 64, 64)

    x1 = torch.Tensor(input)
    x2 = Tensor(input)
    x1.requires_grad = True

    n1 = nn.MaxPool2d(k_size, s, p, d)
    n2 = MaxPool2d(k_size, s, p, d)

    y1 = n1(x1)
    y2 = n2(x2)

    assert y1.shape == y2.shape
    assert np.allclose(y1.detach().numpy(), y2)

    loss = y1.sum()
    loss.backward()

    y2.backward(np.ones(y2.shape))

    assert np.allclose(x1.grad.numpy(), x2.grad)


@pytest.mark.parametrize('batch', (2 ** np.arange(8)).tolist())
@pytest.mark.parametrize('in_size', (2 ** np.arange(3, 12)).tolist())
@pytest.mark.parametrize('out_size', (2 ** np.arange(3, 12)).tolist())
def test_linear(batch, in_size, out_size):
    input = np.random.rand(batch, in_size)

    x1 = torch.Tensor(input)
    x2 = Tensor(input)
    x1.requires_grad = True

    n1 = nn.Linear(in_size, out_size)
    n2 = Linear(in_size, out_size)
    n2.weight[:] = n1.weight.data.numpy()
    n2.bias[:] = n1.bias.data.numpy()

    y1 = n1(x1)
    y2 = n2(x2)

    assert y1.shape == y2.shape
    assert np.allclose(y1.detach().numpy(), y2)


    loss = y1.sum()
    loss.backward()

    y2.backward(np.ones(y2.shape))

    assert np.allclose(x1.grad.numpy(), x2.grad)
    assert np.allclose(n1.weight.grad.numpy(), n2.weight.grad)
    assert np.allclose(n1.bias.grad.numpy(), n2.bias.grad)



@pytest.mark.parametrize('seed', list(range(1000)))
def test_loss(seed):
    torch.random.manual_seed(seed)
    input = np.random.randn(64, 10) * 100
    target = np.random.randint(0, 10, 64)

    x1 = torch.Tensor(input)
    x2 = Tensor(input)
    x1.requires_grad = True

    t1 = torch.Tensor(target).long()
    t2 = Tensor(target)

    loss1 = F.cross_entropy(x1, t1, reduction='sum') / 64
    loss2, grad = cross_entropy_loss(x2, t2)

    assert np.allclose(loss1.detach().numpy(), loss2)

    loss1.backward()
    x2.backward(grad)

    assert np.allclose(x1.grad.numpy(), x2.grad)
