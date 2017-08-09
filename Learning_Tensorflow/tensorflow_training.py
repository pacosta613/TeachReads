# https://www.tensorflow.org/get_started/get_started
import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Tensorflow provides optimizers that slowly change each varibale in order to minimize the loss
# The simplest of these is Gradient Descent, it modeifies each variable according to the magintude of
# the derivative of loss with respect to that variable. 0.01 is the learning rate
optimizer = tf.train.GradientDescentOptimizer(0.01)
# I think all the nodes are being tracked so "train" has all the information of the previous nodes
# This allows it to recalculate the loss during each iteration.
train = optimizer.minimize(loss)

# We run the program 1000 times and slowly modify W and b to minimize the loss
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

# These output values are the values that gave the closest results, they minimized the loss
print(sess.run([W, b]))
