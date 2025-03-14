#generate points inside circles as train data 
#demonstrate backpropgation

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_layers=[3, 4, 3], output_size=4, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Initialize weights and biases using He Initialization
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

    def softmax_derivative(self, softmax_output, target_one_hot):
        return softmax_output - target_one_hot

    def cross_entropy_loss(self, predictions, targets):
        m = targets.shape[1]  # Number of samples
        return -np.sum(targets * np.log(predictions + 1e-9)) / m  # Avoid log(0)

    def forward_backward_propagation(self, X, Y, cur_learning_rate):
        # Ensure that Y is an integer array for one-hot encoding
        Y = Y.astype(int)

        activations = [X]  # Store activations
        zs = []  # Store pre-activation values

        # Forward through ReLU hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            zs.append(z)
            activation = self.relu(z)
            activations.append(activation)

        # Forward through Softmax output layer
        z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        zs.append(z)
        activation = self.softmax(z)
        activations.append(activation)

        # Compute Loss
        one_hot_Y = np.eye(self.layer_sizes[-1])[Y].T  # Convert to one-hot encoding
        loss = self.cross_entropy_loss(activation, one_hot_Y)

        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)

        # Backpropagate output layer (Softmax Backpropagation)
        delta = self.softmax_derivative(activation, one_hot_Y)
        gradients_w[-1] = np.dot(delta, activations[-2].T) / Y.shape[0]
        gradients_b[-1] = np.sum(delta, axis=1, keepdims=True) / Y.shape[0]

        # Backpropagate hidden layers (ReLU Backpropagation)
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(self.weights[i+1].T, delta) * self.relu_derivative(zs[i])
            gradients_w[i] = np.dot(delta, activations[i].T) / Y.shape[0]
            gradients_b[i] = np.sum(delta, axis=1, keepdims=True) / Y.shape[0]

        # Gradient Descent Update
        for i in range(len(self.weights)):
            self.weights[i] -= cur_learning_rate * gradients_w[i]
            self.biases[i] -= cur_learning_rate * gradients_b[i]

        return loss

    def train(self, X, Y):
        cur_learning_rate = self.learning_rate
        minimum_learning_rate = self.learning_rate * 0.0001

        epsilonStall = 1e-8  # Small threshold to determine convergence
        countThreshold = 100
        countStall =  0
        last_loss = -1
        


        for epoch in range(self.epochs):

            loss = self.forward_backward_propagation(X, Y, cur_learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Learning rate: {cur_learning_rate:.6f}")

            if (abs(loss - last_loss) < epsilonStall):
                countStall += 1
                
            if countStall > countThreshold:
                countStall = 0

                cur_learning_rate = cur_learning_rate * 0.1
                if cur_learning_rate < minimum_learning_rate:
                    break

            last_loss = loss

    def predict(self, X):
        activation = X
        for i in range(len(self.weights) - 1):
            activation = self.relu(np.dot(self.weights[i], activation) + self.biases[i])
        activation = self.softmax(np.dot(self.weights[-1], activation) + self.biases[-1])
        return np.argmax(activation, axis=0)

# Generate circular training data with specified diameters
def generate_circle_data_edge(radius, num_points, center, label):
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    labels = np.full(num_points, label)
    return np.array([x, y]), labels


# Modify circular data generation to add points inside the circles as well
def generate_circle_data_within(radius, num_points, center, label):
    """Generate points inside the circle, including the center."""
    # Uniformly distribute points inside the circle
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.sqrt(np.random.uniform(0, radius**2, num_points))  # Radial distance from center
    x = center[0] + r * np.cos(angles)
    y = center[1] + r * np.sin(angles)
    labels = np.full(num_points, label)
    return np.array([x, y]), labels


def generate_train_data():
        
    # Create circular data for each class with points inside and outside of the circles (2, 4, 5, 10)
    num_points_per_circle = 1000

    circle_centers = [(0, 0), (6, 6), (4, 18), (20, 20)]  # Different centers for each circle
    #circle_centers = [(0, 0), (6, 6), (10, 4), (20, 20)]  # Different centers for each circle

    diameters = [2, 4, 5, 10]

    data_X = np.empty((2, 0))  # To hold all data points
    data_Y = np.empty((0,))  # To hold all labels

    # Generate data for each class, adding points inside the circles as well
    for i, (radius, center) in enumerate(zip(diameters, circle_centers)):
        X_new, Y_new = generate_circle_data_within(radius, num_points_per_circle, center, i)
        data_X = np.hstack((data_X, X_new))
        data_Y = np.concatenate((data_Y, Y_new))

    # Ensure that data_Y is cast to integer before one-hot encoding
    return data_X, data_Y.astype(int)


# Plot the updated decision boundary with circular data
def plot_decision_boundary(nn, X, Y, test_points = None, test_colors = None):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    predictions = nn.predict(grid_points)
    predictions = predictions.reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.3, cmap='viridis')
    plt.scatter(X[0, :], X[1, :], c=Y, edgecolors='k', cmap='viridis')

    if (test_points is not None) and (test_colors is not None):
        plt.scatter(test_points[0, :], test_points[1, :], c=test_colors, edgecolors='k', marker='o', s=100, label="Test Points")


    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary of Neural Network")
    plt.show()


X_train, Y_train = generate_train_data()

nn = NeuralNetwork(input_size=2, hidden_layers=[3, 4, 3], output_size=4, learning_rate=0.01, epochs=50000)
nn.train(X_train, Y_train)


# Generate some new test data points (outside of the training data)
X_test = np.array([
    [4,   4],   # Point inside circle 2 (radius 4)
    [7,   7],   # Point inside circle 4 (radius 10)
    [14, 14],   # Point between circle 2 and circle 4
    [0,   6],   # Point inside circle 3 (radius 5)
]).T  # Shape (2, 4)

# Predict class labels for the test data
predictions = nn.predict(X_test)

# Print predictions
print("Predictions for new data points:", predictions)

plot_decision_boundary(nn, X_train, Y_train, X_test, predictions)



