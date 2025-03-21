
import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights and biases
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_hidden = np.random.rand(1, self.hidden_size)
        self.bias_output = np.random.rand(1, self.output_size)

    def feedforward(self, X):
        # Calculate hidden layer activations
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Calculate output layer activations
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backpropagate(self, X, Y, learning_rate=0.1):
        # Calculate error (difference between target and predicted output)
        output_error = Y - self.final_output
        
        # Output layer delta (error * derivative of sigmoid)
        output_delta = output_error * sigmoid_derivative(self.final_output)

        # Hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        
        # Hidden layer delta (error * derivative of sigmoid)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, Y, epochs=10000, learning_rate=0.1):
        # Train the network over multiple epochs
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagate(X, Y, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(Y - self.final_output))  # Mean squared error
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage:
if __name__ == "__main__":
    # Sample data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
    Y = np.array([[0], [1], [1], [0]])  # Expected output (XOR)

    # Create the neural network (2 input neurons, 2 hidden neurons, 1 output neuron)
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

    # Train the network
    nn.train(X, Y, epochs=10000, learning_rate=0.1)

    # Test the network after training
    print("\nPredicted Output after training(Expected output (XOR)):")

    O = nn.feedforward(X)
    for input, output in zip(X, O):
        print(f"{input} --> {output}.")
