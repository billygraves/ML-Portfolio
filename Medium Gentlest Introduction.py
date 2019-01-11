#https://medium.com/all-of-us-are-belong-to-machines/the-gentlest-introduction-to-tensorflow-248dc871a224
#Gradient Descent: Like you're going down a rolling plain and trying to find the
#lowest point, it is not clear which direction reduces your height the most
#Gradient descent calculates what move is best and then moves X units
#in that direction

# %%importing tensorflow
import tensorflow as tf
import numpy as np

#Two basic tensorflow components:
# %% establishing those two basic components
#placeholders - an entry point to feed actual data into
x = tf.placeholder(tf.float32, [None,1]) #[variable size, one input]
#Variable - a variable we are trying to find good values for to minimize the
#loss function
W = tf.Variable(tf.zeros([1,1])) #[one output, one feature (house size)]
b = tf.Variable(tf.zeros([1])) # [one feature (house size)]
#The linear model then becomes
product = tf.matmul(x,W)
y = product + b

# %% making a placeholder for the loss function
#We also have to make these placeholder for loss functions
y_ = tf.placeholder(tf.float32, [None, 1]) #[variable size, 1 output]
#loss of reducing RMSE becomes
loss = tf.reduce_mean(tf.square(y_-y))

# %% gradient Descent
#With linear model, loss function, and data, we can begin gradient Descent
#This will minimize the loss function and produce the best W and b
train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(loss)
#The 0.00001 is the size of the 'step' that we take in the direction of the
#steepest gradient each time we step, also called the learning rate

#Now we train!
# %% training
#First, all variables have to be initialized
sess = tf.Session()
init = tf.initialize_all_variables()
#However, even though python is an interpretive language, TF is not read
#by default (for performance reasons) and so the TF session must start
sess.run(init)
steps = 1000
for i in range(steps):
    xs = np.array([[i]])
    ys = np.array([[2*i]])
    #We execute train_step with a loop calling sess.run()
    feed = { x:xs, y_:ys}
    sess.run(train_step, feed_dict = feed)

    print('After %d iterations' % i)
    print('W: %f' % sess.run(W))
    print('b: %f' % sess.run(b))
