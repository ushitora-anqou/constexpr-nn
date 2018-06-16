import chainer
import os
import numpy as np


def get_small_mnist(ndim):
    if not os.path.exists('.cache/small_mnist_train.npz'):
        if not os.path.exists('.cache'):
            os.mkdir('.cache')
        src = chainer.datasets.mnist.get_mnist(ndim=2)
        train = ([], [])
        test = ([], [])
        for t in src[0]:
            train[0].append(t[0][3:25,3:25].reshape(484))
            train[1].append(t[1])
        for t in src[0]:
            test[0].append(t[0][3:25,3:25].reshape(484))
            test[1].append(t[1])

        # write
        np.savez_compressed('.cache/small_mnist_train.npz', x=train[0], y=train[1])
        np.savez_compressed('.cache/small_mnist_test.npz', x=test[0], y=test[1])
    else:
        train = np.load('.cache/small_mnist_train.npz')
        train = (train['x'], train['y'])
        test = np.load('.cache/small_mnist_test.npz')
        test = (test['x'], test['y'])

    if ndim == 3:
        train[0].reshape(-1, 1, 22, 22)
    elif ndim == 2:
        train[0].reshape(-1, 22, 22)
    elif ndim != 1:
        raise ValueError('invalid ndim for small MNIST dataset')

    return chainer.datasets.TupleDataset(*train), chainer.datasets.TupleDataset(*test)


train_sample_num = 15000
test_sample_num = 15000


#mnist = chainer.datasets.get_mnist()
mnist = get_small_mnist(1)

train_sample_size = mnist[0]._datasets[0].shape[1]
test_sample_size = mnist[1]._datasets[0].shape[1]

with open('mnist.cpp', 'w') as fh:
    fh.write('constexpr size_t MNIST_TRAIN_SAMPLE_NUM = {};\n'.format(train_sample_num))
    fh.write('constexpr size_t MNIST_TRAIN_SAMPLE_SIZE = {};\n'.format(train_sample_size))
    fh.write('constexpr size_t MNIST_TEST_SAMPLE_NUM = {};\n'.format(test_sample_num))
    fh.write('constexpr size_t MNIST_TEST_SAMPLE_SIZE = {};\n'.format(test_sample_size))

    fh.write('constexpr float MNIST_TRAIN_SAMPLES_X[MNIST_TRAIN_SAMPLE_NUM][MNIST_TRAIN_SAMPLE_SIZE]= {\n')
    for i in range(0, train_sample_num):
        fh.write('{')
        for j in range(0, train_sample_size):
            fh.write('{},'.format(mnist[0]._datasets[0][i][j]))
        fh.write('},\n')
    fh.write('};\n')

    fh.write('constexpr size_t MNIST_TRAIN_SAMPLES_T[MNIST_TRAIN_SAMPLE_NUM] = {\n')
    for i in range(0, train_sample_num):
        fh.write('{},'.format(mnist[0]._datasets[1][i]))
        if i % 100 == 99:
            fh.write('\n')
    fh.write('};\n')

    fh.write('constexpr float MNIST_TEST_SAMPLES_X[MNIST_TEST_SAMPLE_NUM][MNIST_TEST_SAMPLE_SIZE] = {\n')
    for i in range(0, test_sample_num):
        fh.write('{')
        for j in range(0, test_sample_size):
            fh.write('{},'.format(mnist[1]._datasets[0][i][j]))
        fh.write('},\n')
    fh.write('};\n')

    fh.write('constexpr size_t MNIST_TEST_SAMPLES_T[MNIST_TEST_SAMPLE_NUM] = {\n')
    for i in range(0, test_sample_num):
        fh.write('{},'.format(mnist[1]._datasets[1][i]))
        if i % 100 == 99:
            fh.write('\n')
    fh.write('};\n')
