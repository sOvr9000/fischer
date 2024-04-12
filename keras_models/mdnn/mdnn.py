
import numpy as np
from keras.models import Model



__all__ = ['MDNN']

class MDNN:
    def __init__(self, dnn: Model):
        self.dnn = dnn
        self.mem_capacity = self.dnn.input_shape[1]
        self.mem_index = 0
        self.features = np.zeros((1, *self.dnn.input_shape[1:]))
        self.labels = np.zeros((1, self.mem_capacity, *self.dnn.output_shape[1:]))
    def predict(self, X, observe: bool = True):
        if observe:
            self.observe(X)
        h = np.roll(self.features, self.mem_capacity - 1 - self.mem_index % self.mem_capacity, axis=1)
        pred = self.dnn(h).numpy()
        return pred
    def observe(self, X):
        self.features[np.arange(X.shape[0]), self.mem_index % self.mem_capacity] = X
    def label_previous_feature(self, Y):
        self.labels[np.arange(Y.shape[0]), self.mem_index % self.mem_capacity] = Y
        self.mem_index += 1
    def fit(self, **kwargs):
        X = np.zeros((self.features.shape[0] * self.features.shape[1], *self.features.shape[1:]))
        Y = np.zeros((X.shape[0], *self.labels.shape[2:]))
        for k in range(0, X.shape[0], self.features.shape[0]):
            s = slice(k, k+self.features.shape[0])
            i = k//self.features.shape[0]
            X[s] = np.roll(self.features, self.mem_capacity - 1 - i, axis=1)
            X[s, :-1-i] = 0
            Y[s] = self.labels[:, i, :]
        return self.dnn.fit(X, Y, **kwargs)
    def summary(self):
        self.dnn.summary()
        print(f'Input shape: {self.dnn.input_shape}')
    def compile(self, *args, **kwargs):
        self.dnn.compile(*args, **kwargs)
        self.optimizer = self.dnn.optimizer
        self.loss = self.dnn.loss
