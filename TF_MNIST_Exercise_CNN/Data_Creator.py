import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset initialization:

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255
x_train = x_train.astype("float32") / 255

# Image Data Generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15)
datagen.fit(x_train.reshape(-1, 28, 28, 1))
iter = datagen.flow(x_train.reshape(-1, 28, 28, 1), y_train, batch_size=1)
samples = [next(iter) for i in range(120000)]
x_train_aug, y_train_aug = zip(*samples)
x_train_aug = np.array(x_train_aug).reshape(-1, 28, 28) + np.random.normal(0, 0.02, (120000, 28, 28))
y_train_aug = np.array(y_train_aug)

for (x_a, y_a) in zip(x_train_aug[:5], y_train_aug[:5]):
    plt.title(y_a)
    plt.imshow(x_a)
    plt.show()

#to save it
with open("data/train_augment.pkl", 'wb') as f:
    pkl.dump([x_train_aug, y_train_aug], f)
f.close()

# #to load it
# with open("train.pkl", "r") as f:
#     train_x, train_y = pkl.load(f)


