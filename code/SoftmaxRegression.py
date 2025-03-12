# Softmax Regression is a supervised learning algorithm 
# used for multi-class classification. 

"""
Using softmax in binary classification is unnecessary, as it gives the same result as sigmoid but with extra computation.
If you force a binary logistic regression model to use softmax, it will still work but be computationally inefficient.
"""


import random
import math
import numpy as np


import math

# Multi-class Logistic Regression using Softmax
class SoftmaxRegression:
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
        num_classes = len(set(y))

        # Initialize weights and bias
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


# Example Multi-class Dataset (3 classes: 0, 1, 2)
X = [[1, 2], [1, 4], [2, 3], [5, 6], [6, 8], [7, 7], [10, 10], [9, 8], [8, 9]]
y = [0, 0, 0, 1, 1, 1, 2, 2, 2]

# Train Softmax Regression Model
softmax_reg = SoftmaxRegression()
softmax_reg.fit(X, y)

# Predict on New Data
predictions = softmax_reg.predict([[3, 4], [6, 7], [9, 9]])
print("Softmax Regression Predictions:", predictions)  # Expected output: [0, 1, 2] (class labels)
