from __future__ import print_function


import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f)
    return dict


class Dataset:
    def __init__(self, path):
        self.d = unpickle(path)
        self.num_samples = d['data'].shape[0]
        self.index = list(range(self.num_samples))
        np.random.shuffle(self.index)
        self.i = 0
        

    def get_batch(size):
        if i + size >= self.num_samples:
            i = 0
        data = d['data'][i:i + size, :]
        data = np.reshape(data, [-1, 32, 32, 3])
        labels = d['fine_labels'][i:i + size]
        i += size
        return data, labels
    
    def refresh():
        self.i = 0
        np.random.shuffle(self.index)



