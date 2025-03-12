# Logistic Regression is a supervised learning algorithm 
# used for binary classification (i.e., classifying data into two categories, such as spam vs. not spam, or yes vs. no). 
# Unlike linear regression, which predicts continuous values, logistic regression predicts probabilities.
# Logistic regression is trained using gradient descent, which updates the weights to minimize the error. 
# The error is calculated using the log loss (binary cross-entropy function)

"""

4. Advantages of Logistic Regression
✅ Simple and Efficient - Fast to train and works well for simple problems.
✅ Interpretable - Weights can be analyzed to understand feature importance.
✅ Works Well for Linearly Separable Data - Best suited for datasets that can be separated with a straight decision boundary.

5. Limitations of Logistic Regression
❌ Cannot Model Complex Relationships - Only works well when data is linearly separable.
❌ Sensitive to Outliers - A few extreme values can distort predictions.
❌ Limited to Binary Classification - Cannot directly handle more than two classes (use Softmax Regression for multi-class problems).

Key Takeaways
Logistic Regression is used for binary classification.
Uses the Sigmoid function to map predictions to probabilities between 0 and 1.
Gradient Descent updates weights to minimize the log loss function.
Works best for linearly separable problems but struggles with complex patterns.
"""


"""
Using softmax in binary classification is unnecessary, as it gives the same result as sigmoid but with extra computation.
If you force a binary logistic regression model to use softmax, it will still work but be computationally inefficient.
"""

import random
import math
import numpy as np

# 1. Logistic Regression (Binary Classification)
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
    def activate(self, z):
        return self.sigmoid(z)
        #return self.softmax(z)

    def fit(self, X, y):
        self.weights = [0] * len(X[0])  # Initialize weights to zero
        for _ in range(self.epochs):
            for i in range(len(X)):
                z = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                y_pred = self.activate(z)
                error = y[i] - y_pred
                self.weights = [w + self.learning_rate * error * x for w, x in zip(self.weights, X[i])]
                self.bias += self.learning_rate * error

    def predict(self, X):
        return [1 if self.activate(sum(w * x for w, x in zip(self.weights, x)) + self.bias) >= 0.5 else 0 for x in X]


        

# Example Data
X = [[0, 0], [1, 1], [2, 2], [5, 5], [2, 4]]
y = [1, 1, 1, 1, 0]

# Train and Predict
log_reg = LogisticRegression()
log_reg.fit(X, y)
print("Logistic Regression Prediction:", log_reg.predict([[1.5, 1.5]]))
print("Logistic Regression Prediction:", log_reg.predict([[1.5, 3.5]]))