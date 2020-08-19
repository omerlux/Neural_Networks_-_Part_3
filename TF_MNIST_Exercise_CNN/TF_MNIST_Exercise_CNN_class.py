import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import tensorflow as tf
import tensorboard
from scipy.ndimage import rotate
from tensorflow import keras
from tensorflow.keras import layers
from contextlib import redirect_stdout
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


tf.keras.backend.clear_session()  # For easy reset of notebook state.

""" @:returns tilted x_train"""
def augment(x_train):
    for i, x in enumerate(x_train):
        x_train[i] = rotate(x, np.random.uniform(-20,20), reshape=False)
    return x_train


# Dataset initialization:
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.reshape(10000, 784).astype("float32") / 255

# Saving old data:
x_train_orig = x_train.reshape(60000, 784).astype("float32") / 255
y_train_orig = y_train

# Loading new dataset I've made - tilt, zoom & noise
with open("data/train_augment.pkl", "rb") as f:
    x_train_aug, y_train_aug = pkl.load(f)
# Showing some samples
# for (x_a, y_a) in zip(x_train_aug[:4], y_train_aug[:4]):
#     plt.title(y_a)
#     plt.imshow(x_a)
#     plt.show()

x_train = np.concatenate((x_train_orig, x_train_aug.reshape(-1, 784)))
y_train = np.concatenate((y_train, y_train_aug.reshape(-1)))

# # Old augmentation:
# # Train original:
# x_train_orig = x_train.reshape(60000, 784).astype("float32") / 255
# y_train_orig = y_train
# # Train inverted negative:
# x_train_negative = 1 - x_train_orig + np.random.normal(0, 0.02, 784)  # invert colors
# # Train only noise
# x_train_noise = x_train_orig + np.random.normal(0, 0.02, 784)  # added noise
# x_train_noise = augment(x_train_noise.reshape(-1, 28, 28)).reshape(-1, 784)
#
# x_train = np.concatenate((x_train_orig, x_train_noise, x_train_negative))
# y_train = np.concatenate((y_train, y_train, y_train))


class Net(tf.keras.Model):

    def __init__(self):
        super(Net, self).__init__(name='Net')
        self.log_dir = "logs_net_class/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.model = tf.keras.models.Sequential()
        self.model.add(layers.Reshape((28, 28, 1)))
        self.model.add(layers.Conv2D(128, (5, 5), activation='relu', input_shape=(28, 28, 1),
                                     kernel_initializer='random_normal', bias_initializer='zeros'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(64, (5, 5), activation='relu',
                                     kernel_initializer='random_normal', bias_initializer='zeros'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(512, activation='relu',
                                    kernel_initializer='random_normal', bias_initializer='zeros',
                                    kernel_regularizer=keras.regularizers.l2(0.0001)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(rate=0.2))
        self.model.add(layers.Dense(256, activation='relu',
                                    kernel_initializer='random_normal', bias_initializer='zeros',
                                    kernel_regularizer=keras.regularizers.l2(0.0001)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(rate=0.2))
        self.model.add(layers.Dense(64, activation='relu',
                                    kernel_initializer='random_normal', bias_initializer='zeros',
                                    kernel_regularizer=keras.regularizers.l2(0.0001)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(10))

        # self.inputs = layers.Reshape((28, 28, 1))
        # self.conv1 = layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1),
        #                            kernel_initializer='random_normal', bias_initializer='zeros')
        # self.maxp1 = layers.MaxPooling2D((2, 2))
        # self.conv2 = layers.Conv2D(32, (5, 5), activation='relu',
        #                            kernel_initializer='random_normal', bias_initializer='zeros')
        # self.maxp2 = layers.MaxPooling2D((2, 2))
        # self.flat = layers.Flatten()
        # self.dropout1 = layers.Dropout(rate=0.5)
        # self.dense1 = layers.Dense(512, activation='relu',
        #                            kernel_initializer='random_normal', bias_initializer='zeros',
        #                            kernel_regularizer=keras.regularizers.l2(0.0001))
        # self.dropout2 = layers.Dropout(rate=0.5)
        # self.dense2 = layers.Dense(256, activation='relu',
        #                            kernel_initializer='random_normal', bias_initializer='zeros',
        #                            kernel_regularizer=keras.regularizers.l2(0.0001))
        # self.dropout3 = layers.Dropout(rate=0.5)
        # self.dense3 = layers.Dense(64, activation='relu',
        #                            kernel_initializer='random_normal', bias_initializer='zeros',
        #                            kernel_regularizer=keras.regularizers.l2(0.0001))
        # self.outputs = layers.Dense(10)

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        return self.model(inputs)

        # x = self.inputs(inputs)
        # x = self.conv1(x)
        # x = self.maxp1(x)
        # x = self.conv2(x)
        # x = self.maxp2(x)
        # x = self.flat(x)
        # x = self.dropout1(x)
        # x = self.dense1(x)
        # x = self.dropout2(x)
        # x = self.dense2(x)
        # x = self.dropout3(x)
        # x = self.dense3(x)
        # return self.outputs(x)

    def setup(self, optimizer="default", loss="default"):

        # Compiling the model:
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(  # exponential decay schedule
            initial_learning_rate=1e-3,  # good for RMSProp
            decay_steps=1000,
            decay_rate=0.9)

        if (optimizer == "default"):
            optimizer = keras.optimizers.Adamax(learning_rate=lr_schedule)
        if (loss == "default"):
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

    def train(self,
              batch_size=200,
              epochs=75,
              validation_split=0.1):

        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        history = self.model.fit(x_train, y_train, batch_size=batch_size,
                       epochs=epochs, validation_split=validation_split) #,
                       # callbacks=[tensorboard_callback])

        try:
            os.mkdir(self.log_dir)
        except OSError:
            print("WARNING!")
        with open(self.log_dir + "/history.pkl", 'wb') as f:
            pkl.dump([history.history['accuracy'], history.history['val_accuracy']], f)
        f.close()



    def predict(self):
        # test_scores = self.evaluate(x_test, y_test, verbose=2)
        # print("Test loss:", test_scores[0])
        # print("Test accuracy:", test_scores[1])

        # evaluate the model
        train_score = self.model.evaluate(x_train_orig, y_train_orig, verbose=0)
        test_score = self.model.evaluate(x_test, y_test, verbose=0)

        # Saving model summary
        with open(self.log_dir + '/modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                self.model.summary()
            f.write("\r\n"
                    "========== RESULTS ==========\r\n")
            f.write("Train %s: \t%f" % (self.model.metrics_names[0], train_score[0]) + "\n")
            f.write("Train %s: %.2f%%" % (self.model.metrics_names[1], train_score[1] * 100) + "\n")
            f.write("Test %s: \t\t%f" % (self.model.metrics_names[0], test_score[0]) + "\n")
            f.write("Test %s: \t%.2f%%" % (self.model.metrics_names[1], test_score[1] * 100) + "\n")

        # print("Test %s: %.2f%%" % (self.metrics_names[0], train_score[0] * 100))
        # print("Test %s: %.2f%%" % (self.metrics_names[1], train_score[1] * 100))
        #
        # print("Test %s: %.2f%%" % (self.metrics_names[0], test_score[0] * 100))
        # print("Test %s: %.2f%%" % (self.metrics_names[1], test_score[1] * 100))


if __name__ == '__main__':
    net = Net()
    net.setup()
    net.train()
    net.predict()

    # TODO: add early stopping, add callbacks, track everything for HAIM
    # INSIGHTS FROM MNIST IN PYTORCH AND TESNORFLOW
    # Insight - RELU function is the best for MNIST classification.
    #           Droupout after the CNN layer and before the Output layer - will reduce the accuracy

