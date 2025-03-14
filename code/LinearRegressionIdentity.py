# Linear Regression is a supervised learning algorithm 
# Linear regression predicts a continuous value by finding the best-fitting straight line 
# that describes the relationship between the input 𝑋 (features) and the output 𝑌 (target variable).

# Linear regression is used to model the relationship between independent variables (features) 
# and a dependent variable (target/output) using a straight line.

# Linear regression predicts a continuous value by finding the best-fitting straight line 
# that describes the relationship between the input X (features) and the output Y (target variable).

# y=wx+b
# y = Predicted output (dependent variable)
# x = Input feature (independent variable)
# w = Weight (coefficient/slope) (determines how much 
# b = Bias (intercept) (determines where the line crosses the y-axis)

# Goal: Find the values of w and b that best fit the data.
# Goal: Find w and b that minimize this cost.
# Cost function: We measure how well the line fits the data using the Mean Squared Error (MSE).


import numpy as np
import matplotlib.pyplot as plt

# 1. Linear Regression Model
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def log(self):
        """Print model parameters"""
        if self.weights is not None:
            print("Weights:", [f"{w:.2f}" for w in self.weights])
        print(f"Bias: {self.bias:.2f}")

    def reset(self, weights, bias):
        """Initialize weights and bias"""
        self.weights = np.array(weights, dtype=np.float64)
        self.bias = bias


    def activate(self, inputs):
        return self.identity(inputs)
    
    def derivative(self, outputs):
        return 1
    
    def identity(self, inputs):
        """Identity Activation Function (Linear Activation)"""
        return inputs  # Linear regression uses identity activation


    def mean_squared_error(self, predictions, targets):
        """Compute Mean Squared Error (MSE)"""
        return np.mean((predictions - targets) ** 2)

    def compute_gradients(self, input_vector, target_value):
        # Compute weighted sum (prediction before activation)
        z = np.dot(self.weights, input_vector) + self.bias

        # Apply activation function
        prediction = self.activate(z)

        loss = 0

        # Compute error (difference between prediction and actual value)
        error = prediction - target_value

        gradient = self.derivative(prediction)

        # Update weights (using the derivative of MSE)
        self.weights -= self.learning_rate * error * gradient * np.array(input_vector)
        
        # Update bias
        self.bias -= self.learning_rate * error * gradient

        loss = self.mean_squared_error(prediction, target_value)

        return loss

    def train(self, training_data):
        """Train the model using gradient descent"""
        for epoch in range(self.epochs):
            
            total_loss = 0
            for input_vector, target_value in training_data:
                loss = self.compute_gradients(input_vector, target_value)
                total_loss += loss

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(training_data):.4f}")

    def predict(self, input_vector):
        """Predict the output using the trained model"""
        return np.dot(self.weights, input_vector) + self.bias


# Initialize weights and bias
weights = [0.8, 0.5]
bias = 0

# Generate Training Data (y = 3x1 + 2x2 + noise)
np.random.seed(42)
X_train = np.random.uniform(-5, 5, (50, 2))  # 50 samples, 2 features
y_train = 3 * X_train[:, 0] + 2 * X_train[:, 1] + np.random.randn(50) * 2  # Linear relation with noise

# Convert training data into format [(features, target)]
training_data = list(zip(X_train, y_train))

# Train and Predict
lin_reg = LinearRegression(learning_rate=0.01, epochs=1000)
lin_reg.reset(weights, bias)
lin_reg.train(training_data)

# Test Data
X_test = np.array([
    [3, 2],   # Expected output around 3(3) + 2(2) = 13 + noise
    [0, 0],   # Expected output around 0
    [1, -1],  # Expected output around 3(1) + 2(-1) = 1 + noise
    [-3, -2], # Expected output around -3(3) -2(2) = -13 + noise
    [5, 5],   # Expected output around 3(5) + 2(5) = 25 + noise
    [-5, -4]  # Expected output around -3(5) -2(4) = -23 + noise
])

print("\nPredictions on Test Data:")
y_test = []
for input_vector in X_test:
    predict_value = lin_reg.predict(input_vector)
    y_test.append(predict_value)

    print(f"Input: {input_vector}, Predicted Output: {predict_value:.2f}")



# Plot Training Data and Prediction Line
plt.figure(figsize=(8, 6))

#plot relationship between featuren X1 and y
plt.scatter(X_train[:, 0], y_train, color='blue', label="Training Data") 
plt.scatter(X_test[:, 0], y_test, color='red', label="Test Predictions", marker='x')

plt.xlabel("Feature 1 (x1)")
plt.ylabel("Target (y)")
plt.legend()
plt.title("Linear Regression Predictions")
plt.show()
