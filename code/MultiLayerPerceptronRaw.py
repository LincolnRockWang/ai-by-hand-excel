# A Multi-Layer Perceptron (MultiLayerPerceptron) is a type of artificial neural network that consists of multiple layers.
# bug to fix:  loss is always 0

import random
import math
import numpy as np
import matplotlib.pyplot as plt

# 1. Helper Functions (Matrix Operations)
def dot_product(matrix_a, matrix_b):
    """Computes the dot product of two matrices."""
    return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*matrix_b)] for row_a in matrix_a]

def transpose(matrix):
    """Transposes a given matrix."""
    return list(map(list, zip(*matrix)))

def matrix_addition(matrix_a, matrix_b):
    """Adds two matrices."""
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(matrix_a, matrix_b)]

def scalar_multiply(matrix, scalar):
    """Multiplies a matrix by a scalar."""
    return [[elem * scalar for elem in row] for row in matrix]

def matrix_subtraction(matrix_a, matrix_b):
    """Subtracts matrix B from matrix A."""
    return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(matrix_a, matrix_b)]

# 2. Activation Functions
def relu(matrix):
    """Applies ReLU activation function element-wise."""
    return [[max(0, x) for x in row] for row in matrix]

def relu_derivative(matrix):
    """Derivative of ReLU function (0 for x < 0, 1 for x > 0)."""
    return [[1 if x > 0 else 0 for x in row] for row in matrix]


def softmax(matrix):
    exp_matrix = [[math.exp(x - max(row)) for x in row] for row in matrix]
    sum_exp = [sum(row) for row in exp_matrix]
    return [[x / sum_exp[i] for x in exp_matrix[i]] for i in range(len(exp_matrix))]


def cross_entropy_loss(predictions, targets):
    return -sum(sum(t * math.log(p + 1e-9) for t, p in zip(row_t, row_p)) for row_t, row_p in zip(targets, predictions)) / len(targets)


class MultiLayerPerceptron:
    def __init__(self, input_size=2, hidden_layers=[3, 4], output_size=3, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layer_sizes = [input_size] + hidden_layers + [output_size]

        # Initialize weights and biases
        self.weights = [[[random.uniform(-1, 1) for _ in range(self.layer_sizes[i])] for _ in range(self.layer_sizes[i+1])]
                        for i in range(len(self.layer_sizes)-1)]
        self.biases = [[[0] for _ in range(self.layer_sizes[i+1])] for i in range(len(self.layer_sizes)-1)]

    def forward_backward_propagation(self, X, Y):
        """Performs forward and backward propagation, updates weights."""
        activations = [X]  # Store activations
        zs = []  # Store pre-activation values

        # Forward propagation through hidden layers (ReLU)
        for i in range(len(self.weights) - 1):
            z = matrix_addition(dot_product(self.weights[i], activations[-1]), self.biases[i])
            zs.append(z)
            activation = relu(z)
            activations.append(activation)

        # Forward propagation through output layer (Softmax)
        z = matrix_addition(dot_product(self.weights[-1], activations[-1]), self.biases[-1])
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        # Compute Loss
        loss = cross_entropy_loss(activation, Y)

        # Backpropagation
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)

        # Softmax layer backpropagation
        delta = matrix_subtraction(activation, Y)
        gradients_w[-1] = scalar_multiply(dot_product(delta, transpose(activations[-2])), 1 / len(Y))
        gradients_b[-1] = scalar_multiply(delta, 1 / len(Y))

        # Backpropagate hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = dot_product(transpose(self.weights[i+1]), delta)
            delta = [[d * r for d, r in zip(row_d, row_r)] for row_d, row_r in zip(delta, relu_derivative(zs[i]))]
            gradients_w[i] = scalar_multiply(dot_product(delta, transpose(activations[i])), 1 / len(Y))
            gradients_b[i] = scalar_multiply(delta, 1 / len(Y))

        # Update Weights
        for i in range(len(self.weights)):
            self.weights[i] = matrix_subtraction(self.weights[i], scalar_multiply(gradients_w[i], self.learning_rate))
            self.biases[i] = matrix_subtraction(self.biases[i], scalar_multiply(gradients_b[i], self.learning_rate))

        return loss

    def train(self, X, Y):
        """Trains the neural network."""
        for epoch in range(self.epochs):
            loss = self.forward_backward_propagation(X, Y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        predictions = []
        
        for x in X:
            activation = [x]  # For each point, use it as a single input (column vector)
            for i in range(len(self.weights) - 1):
                activation = relu(matrix_addition(dot_product(self.weights[i], activation), self.biases[i]))
            activation = softmax(matrix_addition(dot_product(self.weights[-1], activation), self.biases[-1]))

            # Add the class with the highest probability
            predictions.append(activation.index(max(activation)))
        
        return predictions
    


# Generate training data with one-hot encoded labels
# num samples is actually num samples per class 
def generate_train_data_raw(num_samples=300):
    random.seed(42)

    
    # Generate random data points for three classes
    class1 = [[random.gauss(2, 0.5) for _ in range(num_samples)], [random.gauss(2, 0.5) for _ in range(num_samples)]]
    class2 = [[random.gauss(-2, 0.5) for _ in range(num_samples)], [random.gauss(-2, 0.5) for _ in range(num_samples)]]
    class3 = [[random.gauss(2, 0.5) for _ in range(num_samples)], [random.gauss(-2, 0.5) for _ in range(num_samples)]]

    # Combine data from all classes into one dataset
    X = [class1[0] + class2[0] + class3[0], class1[1] + class2[1] + class3[1]]
    
    # One-hot encode labels for the classes (0, 1, 2)
    Y = [[1, 0, 0] for _ in range(num_samples)] + [[0, 1, 0] for _ in range(num_samples)] + [[0, 0, 1] for _ in range(num_samples)]

    return X, Y

# num samples is actually num samples per class 
def generate_train_data_numpy(num_samples=300):
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
    
    grid_points = []
    for i in range(len(yy)):  
        for j in range(len(xx[i])):  
            grid_points.append([xx[i][j], yy[i][j]])

    predictions = np.array(nn.predict(grid_points)).reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.3, cmap="viridis")
    plt.scatter(X[0, :], X[1, :], c=np.argmax(Y, axis=1), edgecolors='k', cmap="viridis")

    if (test_points is not None) and (test_colors is not None):
        plt.scatter(test_points[0, :], test_points[1, :], c=test_colors, edgecolors='k', marker='o', s=100, label="Test Points")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("MultiLayerPerceptron Decision Boundary")
    plt.show()


X_train, Y_train = generate_train_data_raw()

mlp = MultiLayerPerceptron(input_size=2, hidden_layers=[3, 4], output_size=3, learning_rate=0.01, epochs=1000)
mlp.train(X_train, Y_train)

X_test = [[1, 1], [-2, -2], [3, -2], [2, 4]]
predictions = mlp.predict(X_test)
print("Predictions for new points:", predictions)

plot_decision_boundary(mlp, np.array(X_train), np.array(Y_train), np.array(X_test).T, predictions)

