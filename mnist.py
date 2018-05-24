import chainer

train_sample_num = 1
train_sample_size = 764
test_sample_num = 1
test_sample_size = 764

mnist = chainer.datasets.get_mnist()
with open('mnist.cpp', 'w') as fh:
    fh.write('constexpr float MNIST_TRAIN_SAMPLES_X[{}][{}] = {{\n'.format(train_sample_num, train_sample_size))
    for i in range(0, train_sample_num):
        fh.write('{')
        for j in range(0, train_sample_size):
            fh.write('{},'.format(mnist[0]._datasets[0][i][j]))
        fh.write('},\n')
    fh.write('};\n')

    fh.write('constexpr size_t MNIST_TRAIN_SAMPLES_T[{}] = {{\n'.format(train_sample_num))
    for i in range(0, train_sample_num):
        fh.write('{},'.format(mnist[0]._datasets[1][i]))
        if i % 100 == 99:
            fh.write('\n')
    fh.write('};\n')

    fh.write('constexpr float MNIST_TEST_SAMPLES_X[{}][{}] = {{\n'.format(test_sample_num, test_sample_size))
    for i in range(0, test_sample_num):
        fh.write('{')
        for j in range(0, test_sample_size):
            fh.write('{},'.format(mnist[1]._datasets[0][i][j]))
        fh.write('},\n')
    fh.write('};\n')

    fh.write('constexpr size_t MNIST_TEST_SAMPLES_T[{}] = {{\n'.format(test_sample_num))
    for i in range(0, test_sample_num):
        fh.write('{},'.format(mnist[1]._datasets[1][i]))
        if i % 100 == 99:
            fh.write('\n')
    fh.write('};\n')
