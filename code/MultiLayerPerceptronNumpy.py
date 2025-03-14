# A Multi-Layer Perceptron (MLP) is a type of artificial neural network that consists of multiple layers.

import numpy as np
import matplotlib.pyplot as plt

class MultiLayerPerceptron:
    def __init__(self, input_size=2, hidden_layers=[3, 4], output_size=3, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layer_sizes = [input_size] + hidden_layers + [output_size]

        # He Initialization for weights and biases
        self.weights = [np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * np.sqrt(2/self.layer_sizes[i]) 
                        for i in range(len(self.layer_sizes)-1)]
        self.biases = [np.zeros((self.layer_sizes[i+1], 1)) for i in range(len(self.layer_sizes)-1)]

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=0)

    def cross_entropy_loss(self, predictions, targets):
        m = targets.shape[1]  # Number of samples
        return -np.sum(targets * np.log(predictions + 1e-9)) / m  # Avoid log(0)

    def forward_backward_propagation(self, X, Y):
        Y = Y.astype(int)

        activations = [X]  # Store activations
        zs = []  # Store pre-activation values

        # Forward propagation through hidden layers (ReLU)
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            zs.append(z)
            activation = self.relu(z)
            activations.append(activation)

        # Forward propagation through output layer (Softmax)
        z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        zs.append(z)
        activation = self.softmax(z)
        activations.append(activation)

        # Compute loss
        one_hot_Y = np.eye(self.layer_sizes[-1])[Y].T  # Convert to one-hot encoding
        loss = self.cross_entropy_loss(activation, one_hot_Y)

        # Backpropagation
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)

        # Softmax layer backpropagation
        delta = activation - one_hot_Y
        gradients_w[-1] = np.dot(delta, activations[-2].T) / Y.shape[0]
        gradients_b[-1] = np.sum(delta, axis=1, keepdims=True) / Y.shape[0]

        # Backpropagate hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(self.weights[i+1].T, delta) * self.relu_derivative(zs[i])
            gradients_w[i] = np.dot(delta, activations[i].T) / Y.shape[0]
            gradients_b[i] = np.sum(delta, axis=1, keepdims=True) / Y.shape[0]

        # Gradient Descent Update
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

        return loss

    def train(self, X, Y):
        for epoch in range(self.epochs):
            loss = self.forward_backward_propagation(X, Y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        activation = X
        for i in range(len(self.weights) - 1):
            activation = self.relu(np.dot(self.weights[i], activation) + self.biases[i])
        activation = self.softmax(np.dot(self.weights[-1], activation) + self.biases[-1])
        return np.argmax(activation, axis=0)

# num samples is actually num samples per class 
def generate_train_data(num_samples=300):
    np.random.seed(42)


    # Generate three classes of data
    class1 = np.random.randn(2, num_samples) * 0.5 + np.array([[2], [2]])
    class2 = np.random.randn(2, num_samples) * 0.5 + np.array([[-2], [-2]])
    class3 = np.random.randn(2, num_samples) * 0.5 + np.array([[2], [-2]])

    X = np.hstack((class1, class2, class3))
    Y = np.array([0] * num_samples + [1] * num_samples + [2] * num_samples)

    return X, Y
    


def plot_decision_boundary(nn, X, Y, test_points = None, test_colors = None):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    predictions = nn.predict(grid_points).reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.3, cmap="viridis")
    plt.scatter(X[0, :], X[1, :], c=Y, edgecolors='k', cmap="viridis")

    if (test_points is not None) and (test_colors is not None):
        plt.scatter(test_points[0, :], test_points[1, :], c=test_colors, edgecolors='k', marker='o', s=100, label="Test Points")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("MultiLayerPerceptron Decision Boundary")
    plt.show()


X_train, Y_train = generate_train_data()

mlp = MultiLayerPerceptron(input_size=2, hidden_layers=[3, 4], output_size=3, learning_rate=0.01, epochs=1000)
mlp.train(X_train, Y_train)

X_test = np.array([[1, 1], [-2, -2], [3, -2]]).T  # Test points
predictions = mlp.predict(X_test)
print("Predictions for new points:", predictions)


plot_decision_boundary(mlp, X_train, Y_train, X_test, predictions)
