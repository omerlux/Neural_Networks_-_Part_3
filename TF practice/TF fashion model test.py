# TensorFlow and tf.keras
import datetime

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# ploting image predictions
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    # taking argmax
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'  # true prediction
    else:
        color = 'red'  # false prediction

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],  # prediction
                                         100 * np.max(predictions_array),  # percentage
                                         class_names[true_label]),  # truth
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')


# Getting mnist data
fashion_mnist = keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
train_x = train_x / 255.0  # normalize the data
test_x = test_x / 255.0  # normalize the data

# Classification from 0 to 9:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Creating the model:
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input
    keras.layers.Dense(128, activation='sigmoid'),  # hidden layer
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='sigmoid'),  # hidden layer
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10),  # outputs
])
model.summary()

# Compiling the model - choosing the training method:
model.compile(  # optimizer is adam
    optimizer='adam',
    # cross entropy for multi-classifications
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # monitoring training and testing steps by accuracy
    metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model:
batch = 25
model.fit(x=train_x, y=train_y, epochs=10, batch_size=batch,
          steps_per_epoch=len(train_x) / batch, use_multiprocessing=True,
          callbacks=[tensorboard_callback])

# Evaluating the model:
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=1,
                                     batch_size=1, use_multiprocessing=True)
print('\nTest accuracy:', test_acc)

# Make predictions:
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])  # creating probabilities for each kind of image
predictions = probability_model.predict(test_x)

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in green and incorrect predictions in red.
num_rows = 6
num_cols = 6
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_y, test_x)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_y)
plt.tight_layout()
plt.show()
