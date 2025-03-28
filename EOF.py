import numpy as np


# standardize the data
def standardize(X):
    """
    Standardize the data by removing the mean and scaling to unit variance.

    X : 2-D array (m x n, m features x n samples)
    Z : Standardized data (m x n)
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    # Avoid division by zero (if std is zero then the feature is constant and X-mean is also zero so std set to 1 doesn't matter)
    std[std == 0] = 1
    return (X - mean) / std, mean, std


# Calculate Cov Matrix
def cov_matrix(X):
    """
    Calculate the covariance matrix of the given data.

    X : 2-D array (m x n, m features x n samples)
    R : Covariance matrix (m x m), R = N^(-1) * X * X^T, where N is the number of samples.
    """
    N = X.shape[1]  # Number of samples
    R = 1 / N * np.dot(X, X.T)
    return R


# Calculate Eigenvalues and Eigenvectors
def eigen_decomposition(X):
    R = cov_matrix(X)
    eigenvalues, eigenvectors = np.linalg.eig(R)
    eigenvalues = np.real(eigenvalues)  # Discard imaginary part
    eigenvectors = np.real(eigenvectors)  # Discard imaginary part
    idx = np.argsort(eigenvalues)[::-1]  # Sort in descending order
    L = eigenvalues[idx]
    E = eigenvectors[:, idx]
    return L, E
