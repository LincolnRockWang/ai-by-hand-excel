# Logistic Regression is a supervised learning algorithm 
# used for binary classification (i.e., classifying data into two categories, such as spam vs. not spam, or yes vs. no). 
# Unlike linear regression, which predicts continuous values, logistic regression predicts probabilities.

import random
import math

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        self.weights = [0] * len(X[0])  # Initialize weights
        for _ in range(self.epochs):
            for i in range(len(X)):
                y_pred = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                error = y[i] - y_pred
                self.weights = [w + self.learning_rate * error * x for w, x in zip(self.weights, X[i])]
                self.bias += self.learning_rate * error

    def predict(self, X):
        return [sum(w * x for w, x in zip(self.weights, x)) + self.bias for x in X]

# Example Data
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

# Train and Predict
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("Linear Regression Prediction:", lin_reg.predict([[6]]))
