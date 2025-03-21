import pickle  # In Python 3, use pickle instead of cPickle
import gzip
import numpy as np
import os

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    """

    # Get the current directory of the Python script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Define the full path to the MNIST dataset
    mnist_file_path = os.path.join(current_directory,'data', 'mnist.pkl.gz')

    # Open the file with gzip
    with gzip.open(mnist_file_path, 'rb') as f:
        # Load the data using pickle
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing (training_data, validation_data, test_data). 
    Format adjusted for neural network code.
    """
    # Get the raw MNIST data
    tr_d, va_d, te_d = load_data()

    # Reshape the data to 784-dimensional arrays for each image
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))  # Convert zip to list

    # Prepare validation and test data in the same format
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))  # Convert zip to list

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))  # Convert zip to list

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
