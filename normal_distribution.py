import numpy as np

SIZE=100000

with open('normal_distribution.cpp', 'w') as fh:
    fh.write('constexpr float RANDOM_NORMAL_DIST_TABLE[] = {\n')
    for i in range(0, SIZE):
        fh.write('{},'.format(np.random.normal()))
        if i % 100 == 99:
            fh.write('\n')
    fh.write('};\n')
