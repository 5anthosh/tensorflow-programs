import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

epoch_size = 100 #hyperparameters
batch_size = 500
learning_rate = 0.01
X = np.load("X_mnist.npy")
Y = np.load("Y_mnist.npy")
X = X/255 # normalization
m = X.shape[0]
permu = list(np.random.permutation(np.arange(0, m)))
X = X[permu]
Y = Y[permu]
X_train, X_test = X[0:50000, :], X[50000:55000, :]# splitting training set
Y_train, Y_test = Y[0:50000, :], Y[50000:55000, :]


def random_minibatches(X, Y, n, seed=0): # function for splitting into minibatches
    m = X.shape[0] # getting number of training examples
    num_mini_batches = int(math.floor(m/n))
    np.random.seed(seed)
    permu = list(np.random.permutation(np.arange(0, m)))
    shuffled_x = X[permu]
    shuffled_y = Y[permu]
    mini_batchs = []
    for i in range(int(num_mini_batches)):
        mini_x = shuffled_x[i*n:(i+1)*n]
        mini_y = shuffled_y[i*n:(i+1)*n]
        mini_batchs.append((mini_x, mini_y))
    if m%n != 0:
        mini_x = shuffled_x[num_mini_batches*n:]
        mini_y = shuffled_y[num_mini_batches*n:]
        mini_batchs.append((mini_x, mini_y))
        num_mini_batches += 1
    return mini_batchs, num_mini_batches



X = tf.placeholder(shape=[None,784], name="input_layer", dtype=tf.float32)
Y = tf.placeholder(shape=[None, 10], name="output_layer", dtype=tf.float32)
W_1 = tf.get_variable(
    shape=(784, 400),
    initializer=tf.contrib.layers.xavier_initializer(),
    dtype=tf.float32,
    name="weight_layer_1",
    regularizer=tf.contrib.layers.l2_regularizer(0.1),
)
b_1 = tf.Variable(
    tf.zeros(shape=[400]),
    dtype=tf.float32,
    name="bias_layer_1"
)
W_2 = tf.get_variable(
    shape=[400, 10],
    initializer=tf.contrib.layers.xavier_initializer(),
    dtype=tf.float32,
    name="weight_layer_2",
    regularizer=tf.contrib.layers.l2_regularizer(0.1),
)
b_2 = tf.Variable(
    tf.zeros(shape=[10]),
    dtype=tf.float32,
    name="bias_layer_1"
)
z_1 = tf.matmul(X, W_1) + b_1
a_1 = tf.nn.relu(z_1, name="non_linear_Activation_function")
z_2 = tf.matmul(a_1, W_2) + b_2
output = tf.nn.softmax(z_2)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=z_2,
        labels=Y
    )
)
seed = 0
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for k in range(epoch_size):
        epoch_cost = 0
        mini_batches, m = random_minibatches(X_train, Y_train, batch_size, seed)
        seed += 1
        costs = []
        for i in range(m):
            batch_x, batch_y = mini_batches[i]
            sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})
            cost_2 = sess.run(cost, feed_dict={X:batch_x, Y:batch_y})
            epoch_cost += cost_2/m
        print("cost for epoch " + str(k)+ " :" + str(epoch_cost))
        costs.append(epoch_cost)
    corect_prediction = tf.equal(tf.argmax(z_2, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(corect_prediction, 'float'))
    print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

