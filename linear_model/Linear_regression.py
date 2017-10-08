import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing  import StandardScaler
learning_rate = 0.01
training_epochs = 1000
sess = tf.Session()

dataset = pd.read_csv("Salary_Data.csv")
X_array = np.asarray(dataset['YearsExperience'])
X_array = X_array.reshape(X_array.shape[0],1)
# X_array = X_array.reshape(X_array.shape[0],1)
Y_array = np.asarray(dataset['Salary']).reshape(X_array.shape[0],1)
st = StandardScaler()
Y_array = st.fit_transform(Y_array)
# Y_array = Y_array.reshape(Y_array.shape[0], 1)
number_s = X_array.shape[0]
w_v = np.random.randn() 
b_v = np.random.randn()
# input layer nodes
X = tf.placeholder(dtype=tf.float32, name="Year_of_experience")
Y = tf.placeholder(dtype=tf.float32, name="salary")
# weight and bias
W = tf.Variable(w_v, name="Weight")
B = tf.Variable(b_v, name="bias")
# kernel
prediction = tf.add(tf.multiply(W,X), B)
loss = tf.square(tf.subtract(Y , prediction))
cost = (tf.reduce_sum(loss))/(2*number_s)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
tain = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess.run(init) # initialing all variables
# running gradient descent
for i in range(training_epochs):
    print(sess.run(cost, {X:X_array, Y:Y_array}))
    for j,k in zip(X_array.tolist(),Y_array.tolist()):
        sess.run(tain, {X:j,Y:k})
co = sess.run(cost, {X:X_array, Y:Y_array})
print("cost after trianing",co)

print("weight",  sess.run(W))
print("Bias", sess.run(B))





