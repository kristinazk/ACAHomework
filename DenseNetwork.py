import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Backprop functionality does not work

class DenseLayer:
    def __init__(self, inp_num, num_neurons):
        self.inp_num = inp_num
        self.num_neurons = num_neurons
        self.weights = np.random.rand(num_neurons, inp_num)
        self.bias = np.ones((1, num_neurons))
        self.gradient = None

    def forward_prop(self, X):
        return X @ self.weights.T + self.bias

    def back_prop(self, X, y, lr, num_iter, deriv=1):
        for _ in range(num_iter):
            print(self.weights.T.shape, X.shape)
            self.gradient = -2 * (X.T @ (y - (X @ self.weights.T + self.bias))) / len(X.shape[0]) @ deriv
            self.weights -= lr * self.gradient
            self.bias -= lr * -2 * (y - X @ self.weights.T + self.bias) / len(X.shape[0]) @ deriv


class DenseNetwork:
    def __init__(self, lr, num_iter, *hl_neuron_num):
        self.lr = lr
        self.num_iter = num_iter
        self.hl_neuron_num = list(hl_neuron_num)
        self.layers = []

    def call(self, X, y=None, training=False):
        self.layers.append(DenseLayer(X.shape[1], self.hl_neuron_num[0]))
        for i in range(len(self.hl_neuron_num) - 1):
            self.layers.append(DenseLayer(self.hl_neuron_num[i], self.hl_neuron_num[i + 1]))

        if training:
            self.layers[-1].back_prop(self.output, y, self.lr, self.num_iter)
            for i in range(reversed(len(self.layers) - 1)):
                self.layers[i - 1].back_prop(self.layers[i - 2].forward_prop(X), self.layers[i].forward_prop(X),
                                             self.lr, self.num_iter, self.layers[i].gradient)

        else:
            self.output = self.layers[0].forward_prop(X)
            for i in range(1, len(self.hl_neuron_num)):
                self.output = self.layers[i].forward_prop(self.output)
            return self.output