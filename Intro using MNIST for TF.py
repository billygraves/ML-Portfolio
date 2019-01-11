from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import random
import IPython
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#first, the TensorFlow implementation then the book implementation
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation = tf.nn.relu),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer = 'adam',
			  loss = 'sparse_categorical_crossentropy',
			  metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 5)
print(model.evaluate(x_test, y_test))