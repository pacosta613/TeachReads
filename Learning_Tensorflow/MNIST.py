# How are images of different sizes compared

# Download and Read data automatically
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# The training tensor has the shape [55000, 784] where 55000 is the number of pictures
# and 784 is the number of pixels in each picture. Each pixel can be given a intensity form
# 0 - 1

# For our labels we use One-hot-vectors is a vector with a 1 in one direction and 0 in all the others.
# Those the tensor for labels takes the shape [55000, 10]

# Softmax Regressions. If you want to assign probablities to an object being one several different things use softmax
# This is because softmax regression gives us a list of values between 0 and 1 that add up to 1

# Softmax has two steps if it add up all the evidence for our input being in a certain class
# And then it turns that evidence into probablities
# To tally up the evidence we do a weighted up of pixel intensities, a negative weight is against the classification
# and positive is for the classification.
# We also add in some extra evidence called a bias
# We then turn the evidence into probablities using the softmax() function

# x is the training data image, we specify that there are 784 pixels and None means that we can
# put in any number of images, we haven't specified a length
x = tf.placeholder(tf.float32, [None, 784])
# Here are the weights and biaes by using tf.zero we initial all the values to zero
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# This line implements the model. tf.matmul() multiples x and W, then we add b then use the
# the softmax function
y = tf.nn.softmax(tf.matmul(x, W) + b)


# Cross entropy determined the loss in a model
# To find the cross entropy we need to know the correct labels
y_ =  tf.placeholder(tf.float32, [None, 10])
# Then we implement the cross-entrophy function
# This function takes the log of y, multiplies it with the corresponding y_ adds all these values up
# and finally reduce_mean computes the mean over all branches
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Each step of the loop, we get a "batch" of one hundred random data points from our training set.
# We run train_step feeding in the batches data to replace the placeholders.
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
