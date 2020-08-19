import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.clear_session()  # For easy reset of notebook state.

# Dataset initialization:
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0  # normalize the data
x_test = x_test / 255.0  # normalize the data

# Classification from 0 to 9:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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


class Net(tf.keras.Model):

    def __init__(self):
        super(Net, self).__init__(name='Net')
        self.inputs = layers.Reshape((28, 28, 1))
        self.conv1a = layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same',
                kernel_initializer='random_normal', bias_initializer='zeros')
        self.batchnorm1a = layers.BatchNormalization()
        self.conv1b = layers.Conv2D(64, (3, 3), activation='relu',
                                    kernel_initializer='random_normal', bias_initializer='zeros')
        self.batchnorm1b = layers.BatchNormalization()
        self.maxp1 = layers.MaxPooling2D((2, 2))

        self.conv2a = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                    kernel_initializer='random_normal', bias_initializer='zeros')
        self.batchnorm2a = layers.BatchNormalization()
        self.conv2b = layers.Conv2D(128, (3, 3), activation='relu',
                                    kernel_initializer='random_normal', bias_initializer='zeros')
        self.batchnorm2b = layers.BatchNormalization()
        self.maxp2 = layers.MaxPooling2D((2, 2))

        self.conv3a = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    kernel_initializer='random_normal', bias_initializer='zeros')
        self.batchnorm3a = layers.BatchNormalization()
        self.conv3b = layers.Conv2D(256, (3, 3), activation='relu',
                                    kernel_initializer='random_normal', bias_initializer='zeros')
        self.batchnorm3b = layers.BatchNormalization()
        self.maxp3 = layers.MaxPooling2D((2, 2))

        self.flat = layers.Flatten()
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense1 = layers.Dense(1024, activation='relu',
                kernel_initializer='random_normal', bias_initializer='zeros',
                kernel_regularizer=keras.regularizers.l2(0.0001))
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(512, activation='relu',
                kernel_initializer='random_normal', bias_initializer='zeros',
                kernel_regularizer=keras.regularizers.l2(0.0001))
        self.outputs = layers.Dense(10)

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.inputs(inputs)
        x = self.conv1a(x)
        x = self.batchnorm1a(x)
        x = self.conv1b(x)
        x = self.batchnorm1b(x)
        x = self.maxp1(x)

        x = self.conv2a(x)
        x = self.batchnorm2a(x)
        x = self.conv2b(x)
        x = self.batchnorm2b(x)
        x = self.maxp2(x)

        x = self.conv3a(x)
        x = self.batchnorm3a(x)
        x = self.conv3b(x)
        x = self.batchnorm3b(x)
        x = self.maxp3(x)

        x = self.flat(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return self.outputs(x)

    def setup(self):
        # Compiling the model:
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(  # exponential decay schedule
            initial_learning_rate=1e-2,  # 1e-3 good for RMSProp, 1e-4 for Adam
            decay_steps=1000,
            decay_rate=0.9)
        self.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Loss function
            optimizer=keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6),  # Optimizer
            metrics=["accuracy"],
        )

    def train(self,
                batch_size=64,
                epochs=20,
                validation_split=0.1):
        log_dir = "logs_net_class_fashion/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_3doubledCNN"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # Training the model:
        self.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[tensorboard_callback])

    def evaluation(self):
        test_scores = self.evaluate(x_test, y_test, verbose=2)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])

    def prediction(self):
        # Make predictions:
        probability_model = tf.keras.Sequential([self,
                                                 tf.keras.layers.Softmax()])  # creating probabilities for each kind of image
        predictions = probability_model.predict(x_test)

        # Plot the first X test images, their predicted labels, and the true labels.
        # Color correct predictions in green and incorrect predictions in red.
        num_rows = 6
        num_cols = 6
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            plot_image(i, predictions[i], y_test, x_test)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            plot_value_array(i, predictions[i], y_test)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    net = Net()
    net.setup()
    net.train()
    net.evaluation()
    net.prediction()