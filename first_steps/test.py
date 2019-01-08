import pickle


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


d = unpickle('/home/rriverag/cifar-100-python/train')
