
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# 1. Logistic Regression (Binary Classification)
class LogisticRegressionSigmoid:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0


    def log(self):
        if self.weights:
            for wgts in self.weights:
                print(f"weights = {wgts:.2f}")

        print(f"bias = {self.bias:.2f}")

    
    def reset(self, weights, bias):
        self.weights = weights
        self.bias = bias


    def sigmoid(self, z):
        # Clip values of z to prevent overflow in exp()
        z = max(min(z, 500), -500)  # Clip between -500 and 500
        return 1 / (1 + math.exp(-z))
    
    def activate(self, z):
        return self.sigmoid(z)
    
    def cross_entropy_loss(self, predictions, target_class):
        # Binary classification: predictions is a single scalar
        epsilon = 1e-15  # Small constant to prevent log(0)
    
        p = max(min(predictions, 1 - epsilon), epsilon)  # Clamp between [epsilon, 1 - epsilon]
        return - (target_class * math.log(p) + (1 - target_class) * math.log(1 - p))
    
    # Compute gradients for updating weights
    def compute_gradients(self, input_vector, target_class):
        
        # Compute weighted sums
        z = sum(w * x for w, x in zip(self.weights, input_vector)) + self.bias
        probabilities = self.activate(z)

        
        # Compute gradient updates for weights and biases
        total_loss = 0
    
        error = target_class - probabilities

        # Update weights
        #self.weights = [w + self.learning_rate * error * x for w, x in zip(self.weights, input_vector)]
        for i in range(len(input_vector)):
            self.weights[i] += self.learning_rate * error * input_vector[i]  
        
        # Update bias
        self.bias += self.learning_rate * error

        total_loss += self.cross_entropy_loss(probabilities, target_class)

        return total_loss
    
    def train(self, training_data):

        for epoch in range(self.epochs):
            for input_vector, target_class in training_data:

                total_loss = 0
                for input_vector, target_class in training_data:
                    loss = self.compute_gradients(input_vector, target_class)
                    total_loss += loss

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss = {total_loss / len(training_data):.4f}")

                

    def predict(self, input_vector):
        probabilities = self.activate(sum(w * x for w, x in zip(self.weights, input_vector)) + self.bias)

        predicted_class = [1 if  probabilities>= 0.5 else 0]

        return predicted_class, probabilities
    

# Initialize weights and biases
weights = [0.8, 0.5]
bias = 0
        

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


# Train and Predict
log_reg = LogisticRegressionSigmoid()
log_reg.reset(weights,bias)

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
    print(f"Predicted Class: {predicted_class}, Probabilities: {probabilities:.2f}")


# Plotting the decision boundary
def plot_decision_boundary(log_reg, training_data, x_range=(-10, 10), y_range=(-10, 10)):
    # Create a meshgrid for plotting decision boundary
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 100), np.linspace(y_range[0], y_range[1], 100))
    Z = np.zeros(xx.shape)
    
    for i in range(xx.shape[0]):
        for j in range(yy.shape[1]):
            point = [xx[i, j], yy[i, j]]
            _, prob = log_reg.predict(point)
            Z[i, j] = prob

    # Plotting the data points and decision boundary
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['b', 'g'])

    # Extracting the first two features (ignoring the others for simplicity)
    X = np.array([np.array([x[0][0], x[0][1]]) for x in training_data])
    y = np.array([x[1] for x in training_data])
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k', cmap=plt.cm.Paired)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(log_reg, training_data)