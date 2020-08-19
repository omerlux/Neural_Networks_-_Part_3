import tensorflow as tf
import matplotlib.pyplot as plt


# # Using Python state
# x = tf.zeros([10, 10])
# x += 2  # This is equivalent to x = x + 2, which does not mutate the original value of x
# print(x)
#
# v = tf.Variable(1.0)
# # Use Python's `assert` as a debugging statement to test the condition
# assert v.numpy() == 1.0
#
# # Reassign the value `v`
# v.assign(3.0)
# assert v.numpy() == 3.0
#
# # Use `v` in a TensorFlow `tf.square()` operation and reassign
# v.assign(tf.square(v))
# assert v.numpy() == 9.0


def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(outputs, model(inputs))
    dW2, dW1, db = t.gradient(current_loss, [model.W2, model.W1, model.b])
    model.W2.assign_sub(learning_rate * dW2)
    model.W1.assign_sub(learning_rate * dW1)
    model.b.assign_sub(learning_rate * db)


class Model(object):
    def __init__(self):
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
        self.W1 = tf.Variable(10.0)
        self.W2 = tf.Variable(3.0)
        self.b = tf.Variable(10.0)

    def __call__(self, x):
        return self.W2 * (x ** 2) + x * self.W1 * x + self.b


# starting here:
model = Model()

TRUE_W2 = 5.0
TRUE_W1 = 4.0
TRUE_b = 3.0
NUM_EXAMPLES = 1000

# synthesize the training data by adding random Gaussian (Normal) noise to the inputs
inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = TRUE_W2 * (inputs ** 2) + inputs * TRUE_W1 + TRUE_b + noise

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.title("Model output and input - Current loss %1.6f" % loss(model(inputs), outputs).numpy())
plt.show()

# Collect the history of W-values and b-values to plot later
Ws2, Ws1, bs = [], [], []
epochs = range(20)
for epoch in epochs:
    Ws2.append(model.W2.numpy())
    Ws1.append(model.W1.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(outputs, model(inputs))

    train(model, inputs, outputs, learning_rate=0.1)
    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')
    plt.title("Epoch %2d: W2=%1.2f W1=%1.2f b=%1.2f, loss=%2.5f" % (epoch, Ws2[-1], Ws1[-1], bs[-1], current_loss))
    plt.show()

# Let's plot it all
plt.plot(epochs, Ws2, 'g',
         epochs, Ws1, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W2] * len(epochs), 'g--',
         [TRUE_W1] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W2', 'W1', 'b', 'True W', 'True b'])
plt.show()
