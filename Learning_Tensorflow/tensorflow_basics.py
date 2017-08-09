#  https://www.tensorflow.org/get_started/get_started
import tensorflow as tf

# constants are values that never changes
node1 = tf.constant(3.0 , tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

# This prints the nodes that would evaluate to 3.0 and 4.0, it doesnt print 3.0 and 4.0
print(node1, node2)

# To actually print 3.0 and 4.0 we have to evaluate the nodes
# and we do that by running a computational graph with a session
sess = tf.Session() # This line will give you a bunch of warnings
print(sess.run([node1, node2]))

# Operations are also nodes, such as an addition node
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))

# Instead of using the add node we can just do node1 + node2
node4 = node3 + node1
print("sess.run(node4): ",sess.run(node4))

# Instead of directly giving a node a constant we can give it a variable known as a placeholder
# And then we enter a value for this placeholder when we are trying to run a session
# placeholder are used to feed in your training data
# (For inputting) Notice we arent actually storing the vaule, we are just inputting it
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

# Variables allow us to add trainable parameters to a graph, these values change as
# as the program is trained. They can represent weights in a neural network
# (For Storing) Notice we are storing the value (the initial value)
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# You must manually initilize Variables
init = tf.global_variables_initializer()
# They aren't actually initialized until we call sess.run(init)
sess.run(init)
# Since x is a place holder we can change it's value for 1-4
# print(sess.run(linear_model, {x:[1,2,3,4]}))

# Now to actually test the accuracy of our model we need a placeholder that contains the actual answers
# and we need to create a loss function
y = tf.placeholder(tf.float32)
# The loss function measures how far the current model is from the real answers
# A standard lose model sums the squares between the models answer and the real answer
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# We can reassign W and b since they are Variables
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

# And then rerun the model, to produce a lost of 0, meaning the model and real answers match perfectly
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
