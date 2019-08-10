# Building NN with NumPy

This repo provide the minimum necessary code to build and train a CNN model from scratch using NumPy solely (basically I just trying to mimic what PyTorch framework do).

3 layer and one loss function is implemented: **Conv2d**, **MaxPool2d**, **Linear** and **cross_entropy**. 
They are heavily relied on `numpy.lib.stride_tricks.as_stride` function (Conv2d, MaxPool2d) and `scipy.sparse` matrix (MaxPool2d), 
and have almost the same functionality as their PyTorch version.
You can check out in the file [nn](nn.py).

## Quick Start: MNIST classification with LeNet

Please download mnist dataset from [here](http://yann.lecun.com/exdb/mnist/).

```
python main.py /path/to/mnist/data
```
I got a 96.30% test accuracy, 97.61% training accuracy and final loss around 0.0772 on my laptop. 
It took about 20 minutes to complete.

## Requirements
* NumPy
* SciPy (for MaxPool2d)

To run the mnist example, some extra package need to install:
* tqdm
* matplotlib
* Scikit-learn

## Notes
Because I focusing on the core part of understanding neural network (forward and gradient back propagation...), the module reusability is limited 
(ex: you can't forward a module twice before backward the loss).
