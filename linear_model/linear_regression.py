import numpy as np


class LinearRegression:

    def __init__(self):
        self.X = 0
        self.Y = 0
        self.w = 0
        self.b = 1
        self.learning_rate = 0.01
        self.lamda = 0
        self.iteration = 1000
        self.use_bias = True
        self.m = 0

    def _cal_cost(self, z):
        cost = 1/(2*self.m)*np.sum(np.square(self.Y - z))\
         + self.lamda*np.sum(np.square(self.w))
        return cost

    def _compute_z(self):
        if self.use_bias:
            z = self.b + np.dot(self.X, self.w)
            return z
        return np.dot(self.X, self.w)

    def _compute_gradient(self, z):
        dw = (1 / self.m) * np.dot(self.X.T, (z - self.Y)) + (self.lamda/self.m)*self.w
        if self.use_bias:
            db = (1/self.m)*np.sum(z - self.Y)
            return dw, db
        return dw

    def _update_parameters(self):
        z = self._compute_z()
        if self.use_bias:
            dw, db = self._compute_gradient(z)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
        else:
            dw = self._compute_gradient(z)
            self.w = self.w - self.learning_rate * dw

    def fit(self, x, y, learning_rate=0.01, lamda=0, epochs=1000, use_bias=True):
        self.X = x
        self.Y = y.reshape((-1, 1))
        self.w = np.random.rand(x.shape[-1], 1)
        self.use_bias = use_bias
        self.m = x.shape[0]
        self.lamda = lamda
        self.iteration = epochs
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        for _ in range(self.iteration):
            self._update_parameters()

    def predict(self, x):
        if self.use_bias:
            z = np.dot(x, self.w) + self.b
            return z
        return np.dot(x, self.w)
