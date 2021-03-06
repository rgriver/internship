from __future__ import print_function


import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


class Dataset:
    def __init__(self, path):
        self.d = unpickle(path)
        self.num_samples = self.d[b'data'].shape[0]
        self.index = list(range(self.num_samples))
        np.random.shuffle(self.index)
        self.i = 0

    def get_batch(self, size=None):
        if size is None:
            size = self.num_samples
        if self.i + size >= self.num_samples:
            self.i = 0
        data = self.d[b'data'][self.i:self.i + size, :]
        data = np.reshape(data, [size, 32, 32, 3])
        data = data / 255.0
        data = data * 2.0
        data -= 1
        labels = self.d[b'fine_labels'][self.i:self.i + size]
        self.i += size
        return data, labels

    def clear_index(self):
        self.i = 0
    
    def refresh(self):
        self.i = 0
        np.random.shuffle(self.index)



