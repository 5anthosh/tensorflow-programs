import numpy as np


class LogisticRegression:
    def __init__(self):
        self.X = 0
        self.Y = 0
        self.w = 0
        self.b = 1
        self.learning_rate = 0
        self.lamda = 0
        self.m = 0
        self.iteration = 0

    def _sigmoid(self, z):
        output = 1/(1 + np.exp(-z))
        return output

    def _cal_cost(self, z):
         cost = (1/self.m)*np.sum(np.dot(-self.Y, np.log(z).T)
                                  - np.dot((1 - self.Y), np.log((1-z)))) + (
                 self.lamda/(2*self.m))*np.sum(np.square(self.w.reshape((-1, 1))))
         return cost

    def _compute_prob(self):
        z = np.dot(self.X, self.w)
        a = self._sigmoid(z)
        return a

    def _compute_gradient(self, z):
        dw = (1/self.m)*np.dot(self.X.T, (z - self.Y)) + (self.lamda/self.m)*self.w
        db = (1/self.m)*np.sum((z - self.Y))
        return dw, db

    def _update_parameters(self):
        a = self._compute_prob()
        dw, db = self._compute_gradient(a)
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def get_parameters(self):
        return {"w": self.w, "b": self.b}

    def fit(self, x, y, learning_rate=0.01, lamda=0, epochs=1000, print_cost=False):
        self.X = x
        print(x.shape, y.shape)
        if y.ndim == 1:
            self.Y = y.reshape((-1, 1))
            self.w = np.random.rand(x.shape[-1], 1)
        else:
            self.Y = y
            self.w = np.random.rand(x.shape[-1], y.shape[-1])
            self.b = np.ones((1, y.shape[-1]))
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.iteration = epochs
        self.m = x.shape[0]
        for i in range(self.iteration):
            self._update_parameters()
            if print_cost:
                print("iteration {0} : {1} cost".format(i + 1, self._cal_cost(self._compute_prob())))

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        a = self._sigmoid(z)
        a = np.ones(a.shape)*(a >= 0.5)
        return a
