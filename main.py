import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from nn import Tensor, cross_entropy_loss
from optim import SGD
from utils import read_images, read_labels
from model import LeNet

parser = argparse.ArgumentParser(description='Training LeNet on MNIST using NumPy.')
parser.add_argument('data_dir', type=str, help='directory to mnist data')

args = parser.parse_args()

epoch = 10
lr = 0.1
momentum = 0.8
batch = 256

train_data = read_images(os.path.join(args.data_dir, 'train-images.idx3-ubyte'))
train_labels = read_labels(os.path.join(args.data_dir, 'train-labels.idx1-ubyte'))
test_data = read_images(os.path.join(args.data_dir, 't10k-images.idx3-ubyte'))
test_labels = read_labels(os.path.join(args.data_dir, 't10k-labels.idx1-ubyte'))

# normalize
train_data = (train_data - train_data.mean((1, 2), keepdims=True)) / train_data.std((1, 2), keepdims=True)
test_data = (test_data - test_data.mean((1, 2), keepdims=True)) / test_data.std((1, 2), keepdims=True)

my_net = LeNet()
optimizer = SGD(my_net.parameters(), lr, momentum)

loss_history = []

epoch_steps = train_data.shape[0] // batch + 1
avg_loss = avg_acc = 0

for e in range(epoch):
    if e and e % 3 == 0:
        optimizer.lr *= 0.1

    train_loss = train_acc = 0
    e_data, e_labels = shuffle(train_data, train_labels)

    with tqdm(total=epoch_steps) as pbar:
        for x, t in zip(np.array_split(e_data, epoch_steps), np.array_split(e_labels, epoch_steps)):
            x = Tensor(x[:, None])
            t = Tensor(t)

            optimizer.zero_grad()

            logits = my_net(x)
            loss, grad = cross_entropy_loss(logits, t)
            acc = accuracy_score(t, logits.argmax(1))

            logits.backward(grad)
            optimizer.step()

            loss_history.append(loss)
            train_loss += loss
            train_acc += acc
            if not avg_loss:
                avg_loss = loss
            else:
                avg_loss *= 0.98
                avg_loss += 0.02 * loss
            avg_acc *= 0.98
            avg_acc += 0.02 * acc

            pbar.set_postfix(loss=avg_loss, acc=avg_acc * 100)
            pbar.update()

    train_loss /= epoch_steps
    train_acc /= epoch_steps
    print("Epoch %d: training loss = %.4f, training acc = %.2f" % (e + 1, train_loss, train_acc * 100))

y = my_net(test_data[:, None]).argmax(1)
print("Accuracy on test data: %.2f" % (accuracy_score(test_labels, y) * 100))

loss_history = np.array(loss_history)
cum_loss = np.cumsum(np.pad(np.array(loss_history), (10, 10), 'edge'))
moving_avg_loss = (cum_loss[11:] - cum_loss[:-11]) / 11
plt.plot(loss_history, label='original loss')
plt.plot(moving_avg_loss, label='smoothed loss')
plt.ylabel('cross entropy')
plt.xlabel('steps')
plt.legend()
plt.show()
