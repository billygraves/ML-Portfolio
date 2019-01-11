# %% imports!
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(tf.__version__)

# %% we are using the fashion MNIST - ten clothing classifications
#60k for training, 10k for testing
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#Returns the two training arrays and the two test arrays
#Images are 28x28 NumPy arrays, with pixels ranging from 0 to 255
#Labels are an array of integers from 0 to 9 such that
# %% class names, giving them acutal labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# %% data exploration is important whenever you work with data!
train_images.shape
#60000x28x28, so number of images by size of each image
len(train_labels) #This better be the same size... it is!
train_labels
test_images.shape
len(test_labels)

# %% data preprocessing
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.show()

# %% rescale the data between 0 and 1, will make it easier for the comp to understand
train_images = train_images / 255.0
test_images = test_images / 255.0

# %% display the first 25 with class names under each
plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# %% Build the model! This is super easy and fun
#basic building block is the layer, especially in tensorflow
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)), #transforms into 1D
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

# %% compile the model
model.compile(optimizer = tf.train.AdamOptimizer(), # how the model updates
            loss = 'sparse_categorical_crossentropy', # measures accuracy during training
            metrics = ['accuracy']) #how we should measure the overall performance

# %% fit the model
model.fit(train_images, train_labels, epochs = 5)

# %% Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)
#With it trained, we can use it to make predictions
predictions = model.predict(test_images)
predictions[0]

#%% sample image
plt.plot()
plt.imshow(test_images[10])
plt.title(class_names[np.argmax(predictions[10])])
plt.show()
print("What the image should have been classified as:",class_names[test_labels[10]])
#This will also display what the image should be named

# %% we can graph all 10 channels to see how it is working overall
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100 * np.max(predictions_array),
                                        class_names[true_label]),
                                        color = color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = '#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
