import numpy as np



def read_images(filename):
    file = open(filename, 'rb')
    s = file.read()
    num_images = int.from_bytes(s[4:8], 'big')
    rows, cols = int.from_bytes(s[8:12], 'big'), int.from_bytes(s[12:16], 'big')
    images = np.frombuffer(s, dtype=np.uint8, offset=16).reshape(num_images, rows, cols)
    return images / 255


def read_labels(filename):
    file = open(filename, 'rb')
    s = file.read()
    num_images = int.from_bytes(s[4:8], 'big')
    labels = np.frombuffer(s, dtype=np.uint8, offset=8)
    assert num_images == len(labels)
    return labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = read_images('/media/ycy/Shared/Datasets/mnist/train-images.idx3-ubyte')
    print(data.shape)
    plt.imshow(data[0])
    plt.show()

    data = read_labels('/media/ycy/Shared/Datasets/mnist/train-labels.idx1-ubyte')
    print(data)

    #print([(i, j) for i, j in zip(range(10), range(10, -1, -1))])
