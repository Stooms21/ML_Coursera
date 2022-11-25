import numpy as np


def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros(num_iters)

	for iter in range(num_iters):



		J_history(iter) = computeCost(X, y, theta)
