from keras.datasets import mnist
from autokeras import ImageClassifier
import tensorflow
import numpy as np


def load_data(path="MNIST_data/mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    print(tensorflow.__version__)
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(x_train.shape+(1,))
    x_test = x_test.reshape(x_test.shape+(1,))
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y * 100)