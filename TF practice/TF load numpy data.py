import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import max_norm

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
  train_x = data['x_train']
  train_y = data['y_train']
  test_x = data['x_test']
  test_y = data['y_test']

# make a dataset from a numpy array
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

# shuffle and batch the datasets
batch_size = 32
shuffle_buffer_size = 100
train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='sigmoid', kernel_constraint=max_norm(2.)),
    tf.keras.layers.Dense(128, activation='sigmoid', kernel_constraint=max_norm(2.)),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])    # for classification of numbers output

# train the model
model.fit(train_dataset, epochs=10, batch_size=batch_size,
          steps_per_epoch=len(train_x) / batch_size, use_multiprocessing=True)

# evaluation
model.evaluate(test_dataset)