"""
Using softmax in binary classification is unnecessary, as it gives the same result as sigmoid but with extra computation.
If you force a binary logistic regression model to use softmax, it will still work but be computationally inefficient.

Key Takeaways
✅ Sigmoid is the correct choice for binary classification because it is computationally simpler.
✅ Softmax works the same way for two classes, but it adds unnecessary complexity.
✅ Softmax is only needed for multi-class classification (3 or more classes).
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# Logistic Regression using Sigmoid
class LogisticRegressionSigmoid:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        self.weights = [0] * len(X[0])  # Initialize weights to zero
        for _ in range(self.epochs):
            for i in range(len(X)):
                z = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                y_pred = self.sigmoid(z)
                error = y[i] - y_pred
                self.weights = [w + self.learning_rate * error * x for w, x in zip(self.weights, X[i])]
                self.bias += self.learning_rate * error

    def predict(self, X):
        return [1 if self.sigmoid(sum(w * x for w, x in zip(self.weights, x)) + self.bias) >= 0.5 else 0 for x in X]

# Logistic Regression using Softmax (for binary classification)
class LogisticRegressionSoftmax:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def softmax(self, z):
        exp_z = [math.exp(i) for i in z]
        sum_exp_z = sum(exp_z)
        return [i / sum_exp_z for i in exp_z]

    def fit(self, X, y):
        num_features = len(X[0])
        num_classes = 2  # Binary classification

        self.weights = [[0] * num_features for _ in range(num_classes)]
        self.bias = [0] * num_classes

        for _ in range(self.epochs):
            for i in range(len(X)):
                z = [sum(w * x for w, x in zip(self.weights[c], X[i])) + self.bias[c] for c in range(num_classes)]
                y_pred = self.softmax(z)

                # One-hot encoding for y
                y_one_hot = [1 if j == y[i] else 0 for j in range(num_classes)]

                # Update weights and biases using gradient descent
                for c in range(num_classes):
                    error = y_one_hot[c] - y_pred[c]
                    self.weights[c] = [w + self.learning_rate * error * x for w, x in zip(self.weights[c], X[i])]
                    self.bias[c] += self.learning_rate * error

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            z = [sum(w * x for w, x in zip(self.weights[c], X[i])) + self.bias[c] for c in range(len(self.weights))]
            y_pred = self.softmax(z)
            predictions.append(y_pred.index(max(y_pred)))  # Class with highest probability
        return predictions


# Example Binary Classification Dataset
X = [[1, 2], [2, 3], [3, 5], [5, 8], [6, 10]]
y = [0, 0, 1, 1, 1]

# Train Logistic Regression with Sigmoid
sigmoid_log_reg = LogisticRegressionSigmoid()
sigmoid_log_reg.fit(X, y)

# Train Logistic Regression with Softmax
softmax_log_reg = LogisticRegressionSoftmax()
softmax_log_reg.fit(X, y)

# Predict on New Data
new_data = [[2, 3], [4, 6], [6, 9]]
sigmoid_predictions = sigmoid_log_reg.predict(new_data)
softmax_predictions = softmax_log_reg.predict(new_data)

# Print Predictions
print("Predictions using Sigmoid:", sigmoid_predictions)
print("Predictions using Softmax:", softmax_predictions)


# Visualization of decision boundaries (Side-by-Side)
def plot_decision_boundaries(model1, model2, X, y, title1, title2):
    X = np.array(X)
    y = np.array(y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predictions for both models
    predictions1 = np.array(model1.predict(grid)).reshape(xx.shape)
    predictions2 = np.array(model2.predict(grid)).reshape(xx.shape)

    # Plot both decision boundaries side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Sigmoid model decision boundary
    axes[0].contourf(xx, yy, predictions1, alpha=0.3)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    axes[0].set_title(title1)
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")

    # Plot Softmax model decision boundary
    axes[1].contourf(xx, yy, predictions2, alpha=0.3)
    axes[1].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    axes[1].set_title(title2)
    axes[1].set_xlabel("Feature 1")

    plt.tight_layout()
    plt.show()


# Plot both decision boundaries side by side
plot_decision_boundaries(sigmoid_log_reg, softmax_log_reg, X, y,
                         "Decision Boundary - Sigmoid Logistic Regression",
                         "Decision Boundary - Softmax Logistic Regression")
