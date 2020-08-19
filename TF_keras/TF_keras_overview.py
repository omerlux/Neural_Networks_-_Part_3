import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Setting the model:
model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu', bias_regularizer=tf.keras.regularizers.l2(0.01),
          kernel_regularizer=tf.keras.regularizers.l2(0.01),
          kernel_initializer=tf.initializers.RandomNormal(stddev=0.01)))
# Add another:
model.add(layers.Dense(64, activation='relu', bias_regularizer=tf.keras.regularizers.l2(0.01),
          kernel_regularizer=tf.keras.regularizers.l2(0.01),
          kernel_initializer=tf.initializers.RandomNormal(stddev=0.01)))
# Add an output layer with 10 output units:
model.add(layers.Dense(10))

# Set-up training:
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# example for numpy data:
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

# model.fit(data, labels, epochs=10, batch_size=32,
#           validation_data=(val_data, val_labels))

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32)

# model.fit(dataset, epochs=10,
#           validation_data=val_dataset)

# Prediction:
# With Numpy arrays
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

# model.evaluate(data, labels, batch_size=32)

# With a Dataset
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

# model.evaluate(dataset)

# result = model.predict(data, batch_size=32)
# print(result.shape)

# ----------------------------------------------------------------
import TF_keras.TF_keras_model_class as km

"""
Create a model using your custom model:
"""

model = km.MyModel(num_classes=10)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)


""" Create a model using your custom layer: """

model = tf.keras.Sequential([km.MyLayer(10)])
# The compile step specifies the training configuration
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=6)

""" Callbacks to save data: """

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))
# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')
# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./weights/my_model')

""" Save just the model configuration: """

json_string = model.to_json()
json_string
import json
import pprint
pprint.pprint(json.loads(json_string))

""" Recreate the model (newly initialized) from the JSON: """
# fresh_model = tf.keras.models.model_from_json(json_string)      # TODO: Problem restoring data from another file
# # Recreate the model from the YAML:
# yaml_string = model.to_yaml()
# print(yaml_string)
# fresh_model = tf.keras.models.model_from_yaml(yaml_string)


""" Save the entire model in one file """
# Create a simple model
model = tf.keras.Sequential([
  layers.Dense(10, activation='relu', input_shape=(32,)),
  layers.Dense(10)
])
model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)


# Save entire model to a HDF5 file
model.save('my_model')

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('my_model')
