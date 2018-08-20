import numpy as np


def layer_sizes(x, h, y):
    n_x = x.shape[0]
    n_h = h
    n_y = y.shape[0]

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):

    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    parameters = {
        "W1": w1,
        "b1": b1,
        "W2": w2,
        "b2": b2,
    }
    return parameters


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def forward_propagation(x, parameters):

    w1 = parameters['W1']
    w2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    cache = {
        "Z1": z1,
        "Z2": z2,
        "A1": a1,
        "A2": a2,
    }
    return a2, cache


def compute_cost(a2, y):
    m = y.shape[1]

    log_it = y*np.log(a2) + (1 - y)*np.log((1-a2))
    cost = -np.sum(log_it)/m

    return cost


def calculate_derivatives(parameters, cache, x, y):
    m = x.shape[1]
    # w1 = parameters['W1']
    w2 = parameters['W2']

    a1 = cache['A1']
    a2 = cache['A2']

    dz2 = a2 - y
    dw2 = (1/m)*np.dot(dz2, a1.T)
    db2 = (1/m)*np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1/m)*np.dot(dz1, x.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {
        'dW1': dw1,
        'dW2': dw2,
        'db1': db1,
        'db2': db2,
    }

    return grads


def update_parameters(parameters, grads, learning_rate=0.01):

    w1 = parameters['W1']
    b1 = parameters['b1']
    w2 = parameters['W2']
    b2 = parameters['b2']

    dw1 = grads['dW1']
    db1 = grads['db1']
    dw2 = grads['dW2']
    db2 = grads['db2']

    parameters["W1"] = w1 - learning_rate*dw1
    parameters['b1'] = b1 - learning_rate*db1
    parameters['W2'] = w2 - learning_rate*dw2
    parameters['b2'] = b2 - learning_rate*db2

    return parameters


def nn_model(x, y, n_h, num_iterations=10000, print_cost=False):
    n_x = x[0]
    n_y = y[0]
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_iterations):
        a2, cache = forward_propagation(x, parameters)
        cost = compute_cost(a2, y)
        grads = calculate_derivatives(parameters, cache, x, y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {0}: {1}".format(i, cost))
    return parameters


def predict(parameters, x):
    a2, cache = forward_propagation(x, parameters)
    predictions = np.round(a2)
    return predictions
