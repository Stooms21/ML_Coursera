import numpy as np


def featureNormalize(X):
	X_norm = X
	mu = np.zeros(1, np.size(X, 1))
	sigma = np.zeros(1, np.size(X, 1))