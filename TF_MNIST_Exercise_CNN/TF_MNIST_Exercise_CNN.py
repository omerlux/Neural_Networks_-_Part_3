import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.clear_session()  # For easy reset of notebook state.

# Dataset initialization:
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_invtrain = 1 - x_train.reshape(60000, 784).astype("float32") / 255 + np.random.normal(0, 0.01, 784)
x_train = x_train.reshape(60000, 784).astype("float32") / 255 + np.random.normal(0, 0.01, 784)  # added noise
x_test = x_test.reshape(10000, 784).astype("float32") / 255


# Creating the model:
inputs = keras.Input(shape=(784,))
inputs2 = layers.Reshape((28, 28, 1))(inputs)
# lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
x = layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1),          # out is 24x24
                  kernel_initializer='random_normal', bias_initializer='zeros')(inputs2)
x = layers.MaxPooling2D((2, 2))(x)                                              # out is 12x12
x = layers.Conv2D(40, (5, 5), activation='relu',                                # out is 8x8
                  kernel_initializer='random_normal', bias_initializer='zeros')(x)
x = layers.MaxPooling2D((2, 2))(x)                                              # out is 4x4
x = layers.Flatten()(x)
x = layers.Dropout(rate=0.5)(x)
x = layers.Dense(64, activation='relu',
                 kernel_initializer='random_normal', bias_initializer='zeros')(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="MNIST_model")
keras.utils.plot_model(model, "MNIST_model_scheme.png", show_shapes=True) # Model scheme

# Compiling the model:
lr_schedule = keras.optimizers.schedules.ExponentialDecay(  # exponential decay schedule
    initial_learning_rate=1e-3,  # good for RMSProp
    decay_steps=1000,
    decay_rate=0.9)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),  # Loss function
    optimizer=keras.optimizers.RMSprop(learning_rate=lr_schedule),  # Optimizer
    metrics=["accuracy"],
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_allRelu_wNoise_HalfInvHalfRegular"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training the model:
model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.1, callbacks=[tensorboard_callback])
# 10% is validation

# Evaluation of the model:
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
