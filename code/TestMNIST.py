import random
import numpy as np



# Activation functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(x):
        return np.maximum(0, x)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=0)

def softmax_derivative(z, target_one_hot):
    return z - target_one_hot


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def train(self, training_data, eta, test_data):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for i, (x, y) in enumerate(training_data):
            print("Data {} / {}".format(i, len(training_data)))

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(training_data)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(training_data)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


    def train_by_batch(self, training_data, epochs, epoch_batch_size, eta, test_data):
        n_test = len(test_data)
        n_train = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            epoch_batches = [
                training_data[k:k+epoch_batch_size]
                for k in range(0, n_train, epoch_batch_size)
            ]
            for epoch_batch in epoch_batches:
                self.train_epoch_batch(epoch_batch, eta)

            print("Epoch {}: {} / {}".format(epoch, self.predict(test_data), n_test))

    def train_epoch_batch(self, epoch_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in epoch_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(epoch_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(epoch_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]  # list to store activations layer by layer
        zs = []  # list to store all z vectors layer by layer

        # forward pass
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backpropagate through layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        return (nabla_b, nabla_w)

    def predict(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y



import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# training_data consists of rows of (x,y)
# x is a list of floats(0,1) which represents greycolor value in a 28x28 pixel map
# y is a list of floats(0 or 1) which is actually a one hot vector that represents the target predict(number)

net = Network([784, 100, 10])
net.train_by_batch(training_data, epochs=20, epoch_batch_size=10, eta=3.0, test_data=test_data)
#net.train(training_data, eta=3.0, test_data=test_data)

print("---------")