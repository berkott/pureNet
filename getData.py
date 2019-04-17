import numpy as np
import mnist

class mnistData:
    def __init__(self):
        self.train_images = self.toVector(mnist.train_images())
        self.train_labels = mnist.train_labels()

        self.test_images = self.toVector(mnist.test_images())
        self.test_labels = mnist.test_labels()

    def toVector(self, data):
        return np.true_divide(data.reshape((data.shape[0], data.shape[1] * data.shape[2])), 255)

    def getData(self):
        return self.train_images, self.train_labels, self.test_images, self.test_labels
