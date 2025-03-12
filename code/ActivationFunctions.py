import math
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    #exp_x = np.exp(x) # This seems to be performing as well
    return exp_x / np.sum(exp_x)

def sigmoid(x):
    return 1/(1+np.exp(x))


# Generate input values
x = np.linspace(-5, 5, 100)

# Compute function outputs
relu_output = relu(x)
leaky_relu_output = leaky_relu(x)
softmax_output = softmax(x)  # Softmax applied element-wise
sigmoid_output = sigmoid(x)

# Plot ReLU
plt.figure(figsize=(10, 5))
plt.plot(x, relu_output, label="ReLU", linewidth=2)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# Plot Leaky ReLU
plt.figure(figsize=(10, 5))
plt.plot(x, leaky_relu_output, label="Leaky ReLU (α=0.01)", linewidth=2, color="red")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("Leaky ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# Plot Softmax
plt.figure(figsize=(10, 5))
plt.plot(x, softmax_output, label="Softmax", linewidth=2, color="green")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("Softmax Activation Function")
plt.xlabel("Input")
plt.ylabel("Probability Output")
plt.legend()
plt.grid(True)
plt.show()

# Plot Sigmoid
plt.figure(figsize=(10, 5))
plt.plot(x, sigmoid_output, label="Sigmoid", linewidth=2, color="green")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()