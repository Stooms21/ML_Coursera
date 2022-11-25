# Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#


import numpy as np
import matplotlib.pyplot as plt
import functions as f

# Initialization

# ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

# Load Data
data = np.transpose(np.loadtxt('../Data/ex1data2.txt', delimiter=","))
X = np.transpose(data[0:2])
y = data[2]
m = len(y)

# Print out some data points
print('First 10 examples from the dataset: \n')
print(' x = [#.0f #.0f], y = #.0f \n', X[0:9][:], y[0:9][:])

print('Program paused. Press enter to continue.\n')
wait = input("Press Enter to continue.")

# Scale features and set them to zero mean
print('Normalizing Features ...\n')

X, mu, sigma = f.featureNormalize(X)

# Add intercept term to X
X = np.column_stack((np.ones(m), X))  # Add a column of ones to x


# ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('Running gradient descent ...\n')

# Choose some alpha value
# alpha = [1, 0.1, 0.01, 0.001]
alpha = 0.001
num_iters = 4000

# Init Theta and Run Gradient Descent
theta = np.zeros(3)
# for i in alpha:
#     theta, J_history = f.gradientDescentMulti(X, y, theta, i, num_iters)

# # Plot the convergence graph
#     plt.plot(np.arange(1, len(J_history) + 1), J_history,
#         label='alpha = ' + str(i))
#     plt.legend()
#     plt.xlabel('Number of iterations')
#     plt.ylabel('Cost J')
#     plt.show()

theta, J_history = f.gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(np.arange(1, len(J_history) + 1), J_history,
    label='alpha = ' + str(alpha))
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(' #f \n', theta)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
price = np.array([1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]]) \
    .dot(theta)  # You should change this


# ============================================================

print(['Predicted price of a 1650 sq-ft, 3 br house '
    '(using gradient descent):\n $#f\n'], price)

print('Program paused. Press enter to continue.\n')
wait = input("Press Enter to continue.")

# ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#

# Load Data
data = np.transpose(np.loadtxt('../ex1data2.txt', delimiter=","))
X = np.transpose(data[0:2])
y = data[2]
m = len(y)

# Add intercept term to X
X = np.column_stack((np.ones(m), X))  # Add a column of ones to x

# Calculate the parameters from the normal equation
theta = f.normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(' #f \n', theta)
print('\n')


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = np.array([1, 1650, 3]).dot(theta)  # You should change this


# ============================================================

print(['Predicted price of a 1650 sq-ft, 3 br house '
    '(using normal equations):\n $#f\n'], price)
