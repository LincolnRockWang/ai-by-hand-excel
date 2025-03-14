# Logistic Classification is a supervised learning algorithm 
# used for binary classification (i.e., classifying data into two categories, such as spam vs. not spam, or yes vs. no). 
# Unlike linear regression, which predicts continuous values, logistic regression predicts probabilities.

"""
Should We Call It "Logistic Classification" Instead?
Yes! Logistic Regression is only used for classification
Calling it "Logistic Classification" would make more sense.
But the term "Logistic Regression" is widely used.
So, even though it's technically a classification algorithm, "Logistic Regression" remains the standard name.'
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt

class LogisticClassificationSoftmax:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.biases = 0

    def reset(self, weights, biases):
        self.weights = weights
        self.biases = biases


    def log(self):
        if self.weights:
            for wgts in self.weights:
                print("weights = " + ", ".join([f"{weight:.2f}" for weight in wgts]))

        for bias in self.biases:
            print(f"biases = {bias:.2f}")

        

        
    def activate(self, inputs):
        return self.softmax(inputs)
    
    def derivative(self, softmax_output):
        """Compute Softmax Jacobian Matrix"""
        jacobian = [[0.0 for _ in range(len(softmax_output))] for _ in range(len(softmax_output))]
        for i in range(len(softmax_output)):
            for j in range(len(softmax_output)):
                if i == j:
                    jacobian[i][j] = softmax_output[i] * (1 - softmax_output[i])  # Diagonal terms
                else:
                    jacobian[i][j] = -softmax_output[i] * softmax_output[j]  # Off-diagonal terms
        return jacobian


    def softmax(self, inputs):
        exp_values = [math.exp(x) for x in inputs]
        sum_exp_values = sum(exp_values)
        return [exp_val / sum_exp_values for exp_val in exp_values]



    # Cross-entropy loss function (for both softmax and sigmoid cases)
    def cross_entropy_loss(self, predictions, target_class):
        return -math.log(predictions[target_class])

    # Compute gradients for updating weights
    def compute_gradients(self, input_vector, target_class):
        # Compute weighted sums
        z = [sum(self.weights[class_idx][i] * input_vector[i] for i in range(len(input_vector))) + self.biases[class_idx] for class_idx in range(len(weights))]

        # Apply the chosen activation function
        probabilities = self.activate(z)

        # Compute Softmax Derivative (Jacobian Matrix)
        gradient = self.derivative(probabilities)

        num_classes = len(self.weights)

        # Compute gradient updates for weights and biases
        loss = 0
        for class_idx in range(num_classes):
            # Compute target one-hot vector inside the loop
            target_one_hot = (1 if class_idx == target_class else 0)

            # Compute error for this class
            error = probabilities[class_idx] - target_one_hot

            # Update weights
            for i in range(len(input_vector)):
                self.weights[class_idx][i] -= self.learning_rate * error * gradient[class_idx][i] * input_vector[i]  

            # Update bias
            self.biases[class_idx] -= self.learning_rate * error * gradient[class_idx][i]

            loss += self.cross_entropy_loss(probabilities, target_class)

        return loss

    # Training function
    def train(self, training_data):
        for epoch in range(self.epochs):
            
            total_loss = 0
            for input_vector, target_class in training_data:
                loss = self.compute_gradients(input_vector, target_class)
                total_loss += loss

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(training_data):.4f}")

    # Prediction function
    def predict(self, input_vector):
        z = [sum(self.weights[class_idx][i] * input_vector[i] for i in range(len(input_vector))) + self.biases[class_idx] for class_idx in range(len(self.weights))]

        probabilities = self.activate(z)  # Use general activation function

        predicted_class = probabilities.index(max(probabilities))  # Choose the class with highest probability
        return predicted_class, probabilities

# Initialize weights and biases
weights = [
    [0.8, 0.5],
    [-0.8, -0.5],
]
biases = [0.0, 0.0]

# Example training dataset: [(input_vector, correct_class)]
training_data = [
    ([ 1,  2], 1),
    ([ 2,  3], 1),
    ([ 3,  1], 1),
    ([ 4,  2], 1),
    ([ 5,  3], 1),
    
    ([-1, -2], 0),
    ([-2, -3], 0),
    ([-3, -1], 0),
    ([-4, -2], 0),
    ([-5, -3], 0)
]


log_reg = LogisticClassificationSoftmax(learning_rate=0.01, epochs=1000)
log_reg.reset(weights,biases)

log_reg.log()
log_reg.train(training_data)
log_reg.log()


test_data = [
    ([ 3,  2]),   # Should predict 1
    ([ 0,  0]),   # This is a boundary point (uncertain case)
    ([ 1, -1]),   # Should predict 1
    ([-3, -2]),   # Should predict 0
    ([ 5,  5]),   # Should predict 1
    ([-5, -4])    # Should predict 0
]

for input_vector in test_data:
    predicted_class, probabilities = log_reg.predict(input_vector)
    print(f"Predicted Class: {predicted_class}, Probabilities: {probabilities}")

# Plotting the decision boundary
def plot_decision_boundary(log_reg, training_data, x_range=(-10, 10), y_range=(-10, 10)):
    # Create a meshgrid for plotting decision boundary
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 100), np.linspace(y_range[0], y_range[1], 100))
    Z = np.zeros(xx.shape)
    
    for i in range(xx.shape[0]):
        for j in range(yy.shape[1]):
            point = [xx[i, j], yy[i, j]]
            predicted_class, _ = log_reg.predict(point)  # Only store predicted class index
            Z[i, j] = predicted_class  # Store the class index (not a list)


    # Plotting the data points and decision boundary
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['b', 'g'])

    # Extracting the first two features (ignoring the others for simplicity)
    X = np.array([np.array([x[0][0], x[0][1]]) for x in training_data])
    y = np.array([x[1] for x in training_data])
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k', cmap=plt.cm.Paired)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Classification Decision Boundary')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(log_reg, training_data)