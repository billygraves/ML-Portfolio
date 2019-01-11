#https://www.tensorflow.org/guide/low_level_intro

# %% imports
import pandas as pd
import numpy as np
import tensorflow as tf

# %% not to be run
"""The central unit of data in TensorFlow is the tensor. A tensor consists of
a set of primitive values shaped into an array of any number of dimensions. A
tensor's rank is its number of dimensions, while its shape is a tuple of
integers specifying the array's length along each dimension. Here are some
examples of tensor values:
"""
[1., 2., 3.] # a rank 1 tensor, a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2,3]
[[[1., 2., 3.]], [[7., 8., 9.]]] #a rank 3 tensor with shape [2,1,3]

#Tensor flow comes in two distinct sections,
#1. Building the computational graph (a tf.Graph)
#2. Running the computational graph (using tf.Session())

#The graph is a series of tf objects arranged into a Graph
#Two types of objects
#1. tf.Operation ('ops'): Nodes of the graph, describe the operations that
#   consume and produce tensors.
#2. tf.Tensor(): the edges of the graph, These represent the values that
#   flow through the graphs.

#NOTE: tf.Tensors() do not have values, they are just handles to elements

# %% the most simple is a computational graph
a = tf.constant(3.0, dtype = tf.float32)
b = tf.constant(4.0) #also tf.float32
total = a + b
print(a)
print(b)
print(total)
#note that the value doesn't output what is expected, that's because this only
#estbalishes the graph but does not actually run the computational graph

# %% visualizing the TF graph
#TensorFlow offers a resource called TensorBoard which allows for visualization
#of the graph. This is easily done
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
#This produces an event file in the current directory
#Can launch in terminal by tensorboard --logdir

# %% TF sessions
#to evaluate sessions, instantiate a tf.Session() object, known as a session
#if a tf.graph() is like a python file, a tf.Session() is like the executable

#This starts a tf Session
sess = tf.Session()
print(sess.run(total))
#That gives the expected solution that we were looking for before
#You can also pass multiple tensors in one Session.run()
print(sess.run({'ab':(a,b), 'total': total}))
#During a call, a tensor only has a single value

# %% displaying how tensors only have one value
vec = tf.random_uniform(shape = (3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))
#This shows a different value on each run but consistent values during a single run

# %% Feeding
#Graph is currently not too interesting since it always produces a constant result
#A graph can except internal inputs, known as placeholders, a promise
#to provide a value later
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
print(sess.run(z, feed_dict = {x:3, y:4.5}))
print(sess.run(z, feed_dict = {x: [1, 3], y: [2,4]}))

# %% datasets
#We prefer using data rather than placeholders for streaming data into a model
#to get a runnable tf.Tensor we need to convert it to a tf.data.Iterator
#then call it using tf.data.Iterator.get_next
#The simplest way to get this iterator is with tf.data.Dataset.make_one_shot_iterator

#This is actually a method that I don't see very heavily used
my_data = [[0,1],
            [2,3],
            [4,5],
            [6,7]]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

#If the data depends on the stateful operations you need to initialize the
#iterator before you start using it
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break

# Layers
# %% creating a simple dense layer
x = tf.placeholder(tf.float32, shape = [None, 3])
linear_model = tf.layers.Dense(units = 1)
y = linear_model(x)
"""The layer inspects its input to determine sizes for its internal variables.
So here we must set the shape of the x placeholder so that the layer can build
a weight matrix of the correct size.
Now that we have defined the calculation of the output, y, there is one more
detail we need to take care of before we run the calculation."""
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y, {x: [[1,2,3], [4,5,6]]}))

# %% Layer function shortcuts
x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.layers.dense(x, units = 1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1,2,3], [4,5,6]]}))
#Creates and runs the layer in a single call but makes debugging a bit
#more difficult

# %% feature columns
#Easiest way to experiment with columns is the tf.feature_column.input_layer
#function. This only accepts dense columns, so the result must be wrapped in
#tf.feature_column.indicator_column
features = {
    'sales': [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}
department_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column #would do the procedure above but already done
    ]

inputs = tf.feature_column.input_layer(features, columns)
#Running the inputs tensor will parse features into a batch of vectors
#Feature columns have an internal state, so they often need to be initialized
#Categorical columns use tf.contrib.lookup internally and these require
#tf.tables_initializer
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
#Once initialized, it can be run like any other tensor
sess.run((var_init, table_init))
print(sess.run(inputs))

#Now we can train our own small model!
# %% training
#First, define some inputs
x = tf.constant([[1], [2], [3], [4]], dtype = tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype = tf.float32)
#Next, define the linear_model
linear_model = tf.layers.Dense(units = 1)
y_pred = linear_model(x)
#Evaluate the prediction as follows
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y_pred))
#Note: random values (relatively) and the model hasn't been trained!

# %% LOSS
#To optimize a model, we need a loss function
#Using Mean squared error, standard loss for regression
loss = tf.losses.mean_squared_error(labels = y_true, predictions = y_pred)
print(sess.run(loss))

# %% training
#TensorFlow provides optimizers implementing standard optimization algorithms
#These are subclasses of tf.train.Optimizer, incrimentally changing variables
#in order to minimize loss. The simplest is gradient descent, implemented by
#tf.train.GradientDescentOptimizer
#   it modifies each value according to the magnitude of the derivative of loss
#   with respect to that value. an example:
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)
#Since train is an op, not a tensor, it doesn't produce an output
#We need to run them both at the same point to see the loss at iterations
