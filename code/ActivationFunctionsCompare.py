# Implementing basic machine learning algorithms without third-party libraries
# Logistic Regression, Linear Regression, K-Means Clustering, and PCA
# Testing algoriths on differnt activation functions

import random
import math

# 1. Logistic Regression (Binary Classification)
class LogisticRegression:
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

# Example Data
X = [[0, 0], [1, 1], [2, 2], [5, 5], [2, 4]]
y = [1, 1, 1, 1, 0]

# Train and Predict
log_reg = LogisticRegression()
log_reg.fit(X, y)
print("Logistic Regression Prediction:", log_reg.predict([[1.5, 1.5]]))
print("Logistic Regression Prediction:", log_reg.predict([[1.5, 3.5]]))

# 2. Linear Regression
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


# 3. K-Means Clustering (Unsupervised Learning)
class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []

    def fit(self, X):
        self.centroids = random.sample(X, self.k)
        for _ in range(self.max_iterations):
            clusters = {i: [] for i in range(self.k)}
            for point in X:
                distances = [sum((c - p) ** 2 for c, p in zip(centroid, point)) for centroid in self.centroids]
                cluster_idx = distances.index(min(distances))
                clusters[cluster_idx].append(point)
            new_centroids = [self.compute_centroid(clusters[i]) for i in range(self.k)]
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids

    def compute_centroid(self, points):
        return [sum(p[i] for p in points) / len(points) for i in range(len(points[0]))] if points else [0] * len(points[0])

    def predict(self, X):
        return [self.closest_centroid(x) for x in X]

    def closest_centroid(self, point):
        distances = [sum((c - p) ** 2 for c, p in zip(centroid, point)) for centroid in self.centroids]
        return distances.index(min(distances))

# Example Data
X = [[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]

# Train and Predict
kmeans = KMeans(k=2)
kmeans.fit(X)
print("K-Means Cluster Assignments:", kmeans.predict(X))


# 4. Principal Component Analysis (PCA)
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit_transform(self, X):
        self.mean = [sum(col) / len(col) for col in zip(*X)]
        centered_X = [[x - m for x, m in zip(row, self.mean)] for row in X]

        covariance_matrix = [[sum(a * b for a, b in zip(col1, col2)) / len(X) for col2 in zip(*centered_X)]
                             for col1 in zip(*centered_X)]

        eigenvalues, eigenvectors = self.eigen_decomposition(covariance_matrix)
        sorted_indices = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)
        self.components = [eigenvectors[i] for i in sorted_indices[:self.n_components]]

        return [[sum(a * b for a, b in zip(row, component)) for component in self.components] for row in centered_X]

    def eigen_decomposition(self, matrix):
        from functools import reduce
        def identity(n): return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        def mat_mult(A, B): return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

        Q = identity(len(matrix))
        R = matrix
        for _ in range(100):  # QR Decomposition Approximation
            Q_new, R = self.qr_decomposition(R)
            Q = mat_mult(Q, Q_new)
        eigenvalues = [R[i][i] for i in range(len(R))]
        eigenvectors = [[Q[j][i] for j in range(len(Q))] for i in range(len(Q[0]))]
        return eigenvalues, eigenvectors

    def qr_decomposition(self, matrix):
        import numpy.linalg as la
        Q, R = la.qr(matrix)
        return Q.tolist(), R.tolist()

# Example Data
X = [[2, 8, 4], [3, 7, 5], [5, 6, 8], [8, 4, 6]]

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print("PCA Reduced Data:", X_reduced)
