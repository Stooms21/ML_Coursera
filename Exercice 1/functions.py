import numpy as np
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
	m = len(y)
	J = 0
	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost of a particular choice of theta
	#               You should set J to the cost.
	J = sum(1. / (2. * m) * (theta.dot(np.transpose(X)) - y)**2)
	return J


def computeCostMulti(X, y, theta):
	m = len(y)
	J = 0
	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost of a particular choice of theta
	#               You should set J to the cost.
	J = sum(1. / (2. * m) * (theta.dot(np.transpose(X)) - y)**2)
	return J


def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros(num_iters)

	for iter in range(num_iters):
		# ====================== YOUR CODE HERE ======================
		# Instructions: Perform a single gradient step on the parameter vector
		#               theta.
		#
		# Hint: While debugging, it can be useful to print out the values
		#       of the cost function (computeCost) and gradient here.
		#
		theta -= alpha * 1. / m * (theta.dot(np.transpose(X)) - y).dot(X)
		J_history[iter] = computeCost(X, y, theta)

	return theta, J_history


def gradientDescentMulti(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros(num_iters)

	for iter in range(num_iters):
		# ====================== YOUR CODE HERE ======================
		# Instructions: Perform a single gradient step on the parameter vector
		#               theta.
		#
		# Hint: While debugging, it can be useful to print out the values
		#       of the cost function (computeCost) and gradient here.
		#
		theta -= alpha * 1. / m * (theta.dot(np.transpose(X)) - y).dot(X)
		J_history[iter] = computeCostMulti(X, y, theta)

	return theta, J_history


def featureNormalize(X):
	# ====================== YOUR CODE HERE ======================
	# Instructions: First, for each feature dimension, compute the mean
	#               of the feature and subtract it from the dataset,
	#               storing the mean value in mu. Next, compute the
	#               standard deviation of each feature and divide
	#               each feature by it's standard deviation, storing
	#               the standard deviation in sigma.
	#
	#               Note that X is a matrix where each column is a
	#               feature and each row is an example. You need
	#               to perform the normalization separately for
	#               each feature.
	#
	# Hint: You might find the 'mean' and 'std' functions useful.
	#
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0)
	X_norm = (X - mu) / sigma

	return X_norm, mu, sigma


def plotData(x, y):
	# ====================== YOUR CODE HERE ======================
	# Instructions: Plot the training data into a figure using the
	#               "figure" and "plot" commands. Set the axes labels using
	#               the "xlabel" and "ylabel" commands. Assume the
	#               population and revenue data have been passed in
	#               as the x and y arguments of this function.
	#
	# Hint: You can use the 'rx' option with plot to have the markers
	#       appear as red crosses. Furthermore, you can make the
	#       markers larger by using plot(..., 'rx', 'MarkerSize', 10)
	plt.plot(x, y, "o")
	plt.xlabel("Population")
	plt.ylabel("Revenue")
	plt.show()


def normalEqn(X, y):
	theta = np.zeros(np.size(X, 1))
	# ====================== YOUR CODE HERE ======================
	# Instructions: Complete the code to compute the closed form solution
	#               to linear regression and put the result in theta.
	theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)),
		np.transpose(X)).dot(y)
	return theta


def warmUpExercise():
	A = []
	# ============= YOUR CODE HERE ==============
	# Instructions: Return the 5x5 identity matrix
	#               In octave, we return values by defining which variables
	#               represent the return values (at the top of the file)
	#               and then set them accordingly.
	A = np.eye(5)
	return A
